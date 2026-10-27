Shader "Custom/NeuralWater"
{
    // Rivers + lakes for NeuralTerrain.cs. Same CDLOD grid as the ground, but
    // displaced to the WATER surface (_WaterTex). Water depth = surface - ground,
    // both read from textures, so no camera depth texture is needed (cheap,
    // works on low-end devices). Only quadtree nodes that contain water are drawn.
    Properties
    {
        [Header(Color)]
        _ShallowColor ("Shallow Color", Color) = (0.20, 0.55, 0.55, 1)
        _DeepColor ("Deep Color", Color) = (0.03, 0.14, 0.22, 1)
        _Clarity ("Clarity (m until it looks deep)", Range(0.2, 20)) = 3
        _ShallowAlpha ("Shallow Transparency", Range(0, 1)) = 0.25
        [Header(Surface)]
        _WaveStrength ("Wave Strength", Range(0, 2)) = 1
        _WaveScale ("Wave Size", Range(0.3, 3)) = 1
        _WaveHeight ("Swell Height (m, deep water)", Range(0, 1)) = 0.15
        _RippleScale ("Ripple Size", Range(0.05, 2)) = 0.45
        _RippleStrength ("Ripple Strength", Range(0, 2)) = 0.5
        _FlowSpeed ("River Flow Speed (m/s)", Range(0, 10)) = 2.5
        _Smoothness ("Sun Glint Sharpness", Range(16, 2048)) = 400
        _Reflection ("Sky Reflection", Range(0, 1)) = 0.8
        [Header(Foam)]
        _FoamColor ("Foam Color", Color) = (0.92, 0.95, 0.95, 1)
        _FoamDepth ("Shore Foam Depth (m)", Range(0, 3)) = 0.6
        _RapidsSlope ("Rapids Start Slope", Range(0.02, 1)) = 0.3
        _FoamAmount ("Foam Amount", Range(0, 1)) = 0.7
    }

    HLSLINCLUDE
    #include "NeuralTerrainCommon.hlsl"

    CBUFFER_START(UnityPerMaterial)
        half4 _ShallowColor;
        half4 _DeepColor;
        float _Clarity;
        float _ShallowAlpha;
        float _WaveStrength;
        float _WaveScale;
        float _WaveHeight;
        float _RippleScale;
        float _RippleStrength;
        float _FlowSpeed;
        float _Smoothness;
        float _Reflection;
        half4 _FoamColor;
        float _FoamDepth;
        float _RapidsSlope;
        float _FoamAmount;
    CBUFFER_END

    static const float2 WIND = float2(0.8, 0.6);

    // Four travelling waves fanned around a direction. Lakes: they drift with the
    // wind at real water-wave speed. Rivers: they are carried downstream by the flow.
    // Returns height in .x and the surface slope (dh/dx, dh/dz) in .yz.
    float3 Waves(float2 xz, float2 dir, float flowSpeed, float t)
    {
        const float4 angle = float4(0.0, 0.55, -0.45, 1.2);
        const float4 len   = float4(9.0, 5.3, 3.1, 1.7);
        const float4 steep = float4(0.10, 0.09, 0.07, 0.05);
        float3 r = 0;
        [unroll] for (int i = 0; i < 4; i++)
        {
            float s, c;
            sincos(angle[i], s, c);
            float2 d = float2(dir.x * c - dir.y * s, dir.x * s + dir.y * c);
            float k = 6.2831853 / (len[i] * _WaveScale);
            float speed = sqrt(9.81 / k) * 0.6 + flowSpeed;   // deep-water wave speed + current
            float phase = k * (dot(d, xz) - speed * t);
            float a = steep[i] / k;                           // amplitude from steepness
            sincos(phase, s, c);
            r.x += a * s;
            r.yz += a * k * c * d;
        }
        return r;
    }
    ENDHLSL

    SubShader
    {
        Tags { "RenderType"="Transparent" "RenderPipeline"="UniversalPipeline" "Queue"="Transparent-10" }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode"="UniversalForward" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            Cull Back

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile_fragment _ _SHADOWS_SOFT
            #pragma multi_compile_fog

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes { float4 positionOS : POSITION; UNITY_VERTEX_INPUT_INSTANCE_ID };
            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float3 positionWS : TEXCOORD0;
                float  fog        : TEXCOORD1;
            };

            Varyings vert(Attributes IN)
            {
                UNITY_SETUP_INSTANCE_ID(IN);
                Varyings OUT;
                float2 xz = CdlodWorldXZ(IN.positionOS.xyz);
                float surface = SampleWater(xz);
                // Gentle swell on deep water only (fades to 0 at the shore so the waterline stays put).
                float deep = saturate((surface - SampleGround(xz)) / 3.0);
                float swell = Waves(xz * 0.35, WIND, 0.0, _Time.y).x * deep * _WaveHeight / 0.35;
                OUT.positionWS = float3(xz.x, _TerrainBaseY + surface + swell, xz.y);
                OUT.positionCS = TransformWorldToHClip(OUT.positionWS);
                OUT.fog = ComputeFogFactor(OUT.positionCS.z);
                return OUT;
            }

            // Gradient of value noise -> ripple normal offset.
            float2 NoiseGrad(float2 p)
            {
                const float d = 0.15;
                float c = NT_ValueNoise(p);
                return float2(NT_ValueNoise(p + float2(d, 0)) - c, NT_ValueNoise(p + float2(0, d)) - c) / d;
            }

            // Fine ripples carried downstream at the real flow speed (m/s). Two copies,
            // half a cycle apart, are cross-faded so the pattern never stretches out.
            float2 FlowRipples(float2 xz, float2 flow)
            {
                const float period = 2.0;
                float t = _Time.y / period;
                float p0 = frac(t), p1 = frac(t + 0.5);
                float w0 = 1.0 - abs(1.0 - 2.0 * p0);
                float2 uv0 = (xz - flow * p0 * period) * _RippleScale;
                float2 uv1 = (xz - flow * p1 * period) * _RippleScale + 3.7;
                float2 a = NoiseGrad(uv0) + NoiseGrad(uv0 * 2.3 + 7.1) * 0.5;
                float2 b = NoiseGrad(uv1) + NoiseGrad(uv1 * 2.3 + 1.3) * 0.5;
                return lerp(b, a, w0);
            }

            // Wind waves carried downstream by the river current (flow-map cross-fade:
            // wave directions stay fixed, so the pattern never twists or tears).
            float3 FlowWaves(float2 xz, float2 flow)
            {
                const float period = 3.0;
                float t = _Time.y / period;
                float p0 = frac(t), p1 = frac(t + 0.5);
                float w0 = 1.0 - abs(1.0 - 2.0 * p0);
                float3 a = Waves(xz - flow * p0 * period, WIND, 0.0, _Time.y);
                float3 b = Waves(xz - flow * p1 * period + 17.3, WIND, 0.0, _Time.y);
                return lerp(b, a, w0);
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float2 xz = IN.positionWS.xz;
                float surface = SampleWater(xz);
                float depth = surface - SampleGround(xz);
                clip(depth - 0.02);

                // Surface slope -> flow direction (downhill) and speed.
                float e = SampleSpacing();
                float ge = e * 4.0; // wide sampling = smooth, stable flow direction
                float2 grad = float2(SampleWater(xz + float2(ge, 0)) - SampleWater(xz - float2(ge, 0)),
                                     SampleWater(xz + float2(0, ge)) - SampleWater(xz - float2(0, ge))) / (2.0 * ge);
                float slope = length(grad);
                float2 flowDir = slope > 0.002 ? -grad / slope : normalize(float2(0.8, 0.6)); // lakes: gentle wind drift
                float speed = _FlowSpeed * (0.15 + saturate(slope * 20.0));
                float2 flow = flowDir * speed;

                // Fine detail fades out with distance so far water stays calm instead of sparkling.
                float dist = distance(_WorldSpaceCameraPos.xyz, IN.positionWS);
                float nearDetail = saturate(1.0 - dist / 500.0);

                float calm = saturate(depth * 2.0); // flatter right at the waterline
                float3 waves = FlowWaves(xz, flow);
                float3 swell = Waves(xz * 0.35, WIND, 0.0, _Time.y);
                float2 ripple = FlowRipples(xz, flow) * 0.15 * _RippleStrength;
                float2 waveSlope = (waves.yz * _WaveStrength * (0.35 + 0.65 * nearDetail) + ripple * nearDetail) * calm
                                 + swell.yz * _WaveHeight * saturate(depth / 3.0);
                float3 n = normalize(float3(-grad.x - waveSlope.x, 1.0, -grad.y - waveSlope.y));

                float3 v = normalize(GetWorldSpaceViewDir(IN.positionWS));
                Light light = GetMainLight(TransformWorldToShadowCoord(IN.positionWS), IN.positionWS, half4(1, 1, 1, 1));
                half3 ambient = SampleSH(float3(0, 1, 0));
                half3 sun = light.color * light.shadowAttenuation;

                // Body color: shallow -> deep by depth.
                float absorb = exp(-depth / _Clarity);
                half3 body = lerp(_DeepColor.rgb, _ShallowColor.rgb, absorb) * (ambient + sun * saturate(light.direction.y) * 0.6);

                // Sky reflection (fresnel) + sun glint.
                float ndv = saturate(dot(n, v));
                float fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
                half3 sky = GlossyEnvironmentReflection(reflect(-v, n), 0.05, 1.0);
                half3 color = lerp(body, sky, fresnel * _Reflection);
                float3 hv = normalize(light.direction + v);
                color += sun * pow(saturate(dot(n, hv)), _Smoothness) * 3.0 * (0.3 + 0.7 * nearDetail);

                // Foam: thin line at the shore, streaks on steep fast water (rapids).
                // Pattern is advected along the flow; far away it averages out to a soft tint.
                float2 foamUV = (xz - flow * _Time.y) * 0.22;
                float pattern = smoothstep(0.4, 0.75, NT_ValueNoise(foamUV) * 0.65 + NT_ValueNoise(foamUV * 2.7 + 5.3) * 0.35);
                pattern = lerp(0.35, pattern, nearDetail);
                float shoreFoam = 1.0 - smoothstep(0.0, _FoamDepth, depth);
                float rapids = smoothstep(_RapidsSlope, _RapidsSlope * 2.0, slope);
                float foam = saturate((shoreFoam * 0.8 + rapids) * pattern) * _FoamAmount;
                color = lerp(color, _FoamColor.rgb * (ambient + sun), foam);

                float alpha = lerp(_ShallowAlpha, 1.0, 1.0 - absorb);
                alpha = saturate(max(alpha, fresnel * _Reflection) + foam);
                alpha *= smoothstep(0.02, 0.15, depth); // soft waterline

                color = MixFog(color, IN.fog);
                return half4(color, alpha);
            }
            ENDHLSL
        }
    }
}
