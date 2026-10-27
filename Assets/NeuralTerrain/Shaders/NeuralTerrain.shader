Shader "Custom/NeuralTerrainCDLOD"
{
    // Ground of the CDLOD terrain (NeuralTerrain.cs).
    // Vertex: one shared grid, instanced per quadtree node, displaced by
    // _HeightTex and morphed between LODs (no pops, no cracks).
    // Fragment: per-pixel normal + curvature from the height texture, biome
    // color (height x moisture), rock on slopes/ridges, snow, wet shores, fog.
    Properties
    {
        [Header(Rock)]
        _RockColor ("Rock Color", Color) = (0.42, 0.40, 0.38, 1)
        _RockSlope ("Rock Slope (start, end)", Vector) = (0.25, 0.45, 0, 0)
        _RidgeRock ("Rock on Ridge Lines", Range(0, 1)) = 0.5
        [Header(Snow)]
        _SnowColor ("Snow Color", Color) = (0.93, 0.95, 0.98, 1)
        _SnowHeight ("Snow Height (0-1)", Range(0, 1)) = 0.78
        _SnowBlend ("Snow Blend", Range(0.001, 0.2)) = 0.05
        [Header(Shores and River Beds)]
        _SandColor ("Shore Sand Color", Color) = (0.76, 0.69, 0.50, 1)
        _ShoreWidth ("Shore Sand Amount", Range(0, 1)) = 0.6
        _WetDarken ("Wet Ground Darkening", Range(0, 1)) = 0.45
        [Header(Detail)]
        _Curvature ("Curvature Shading (crests light, creases dark)", Range(0, 2)) = 0.6
        _ColorNoise ("Color Variation", Range(0, 0.5)) = 0.12
        _Brightness ("Brightness", Range(0.5, 1.5)) = 1
        _Ambient ("Ambient (sky light) Strength", Range(0, 1.5)) = 0.7
    }

    HLSLINCLUDE
    #include "NeuralTerrainCommon.hlsl"

    CBUFFER_START(UnityPerMaterial)
        half4 _RockColor;
        float4 _RockSlope;
        float _RidgeRock;
        half4 _SnowColor;
        float _SnowHeight;
        float _SnowBlend;
        half4 _SandColor;
        float _ShoreWidth;
        float _WetDarken;
        float _Curvature;
        float _ColorNoise;
        float _Brightness;
        float _Ambient;
    CBUFFER_END

    float3 GroundWorldPos(float3 positionOS)
    {
        float2 xz = CdlodWorldXZ(positionOS);
        return float3(xz.x, _TerrainBaseY + SampleGround(xz), xz.y);
    }
    ENDHLSL

    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" "Queue"="Geometry" }

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode"="UniversalForward" }

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
                OUT.positionWS = GroundWorldPos(IN.positionOS.xyz);
                OUT.positionCS = TransformWorldToHClip(OUT.positionWS);
                OUT.fog = ComputeFogFactor(OUT.positionCS.z);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float2 xz = IN.positionWS.xz;
                float e = SampleSpacing();

                // How many height samples this pixel covers -> which mip to read, so
                // lighting stays clean from far away / from above (no dotted aliasing).
                float footprint = max(length(ddx(xz)), length(ddy(xz)));
                float lod = max(0.0, log2(footprint / e));
                float se = e * exp2(lod);

                float h  = SampleGround(xz);
                float hL = SampleGroundLod(xz - float2(se, 0), lod), hR = SampleGroundLod(xz + float2(se, 0), lod);
                float hD = SampleGroundLod(xz - float2(0, se), lod), hU = SampleGroundLod(xz + float2(0, se), lod);
                float3 n = normalize(float3(hL - hR, 2.0 * se, hD - hU));

                // Curvature over a broader area (~4x wider): reads real crests and gullies,
                // not every tiny bump. < 0 on crests, > 0 in creases.
                float ce = se * 4.0, clod = lod + 2.0;
                float cL = SampleGroundLod(xz - float2(ce, 0), clod), cR = SampleGroundLod(xz + float2(ce, 0), clod);
                float cD = SampleGroundLod(xz - float2(0, ce), clod), cU = SampleGroundLod(xz + float2(0, ce), clod);
                float curvature = clamp((cL + cR + cD + cU - 4.0 * SampleGroundLod(xz, clod)) / ce, -1.0, 1.0);

                float h01 = saturate(h / _TerrainParams.w);
                half3 masks = SAMPLE_TEXTURE2D(_MoistureTex, sampler_MoistureTex, MaskUV(xz)).rgb; // moisture, shore, ridge
                float waterDepth = SampleWater(xz) - h;

                // Soft, large-scale variation (patches tens of meters wide) -- reads as natural, never speckled.
                float variation = NT_ValueNoise(xz * 0.006) * 0.55 + NT_ValueNoise(xz * 0.025) * 0.3 + NT_ValueNoise(xz * 0.09) * 0.15 * saturate(1.0 - lod * 0.5);
                half3 albedo = SAMPLE_TEXTURE2D(_BiomeLut, sampler_BiomeLut, float2(h01, saturate(masks.r + (variation - 0.5) * 0.1))).rgb;

                // Sand on shores / river banks (not on steep banks).
                float sand = saturate(masks.g * 1.6 - (1.0 - _ShoreWidth)) * (1.0 - smoothstep(0.35, 0.6, 1.0 - n.y));
                albedo = lerp(albedo, _SandColor.rgb, sand);

                // Rock on steep slopes and along drawn ridge lines.
                float slope = 1.0 - n.y;
                float rock = smoothstep(_RockSlope.x, _RockSlope.y, slope + (variation - 0.5) * 0.08);
                rock = saturate(rock + masks.b * _RidgeRock * smoothstep(0.05, 0.2, slope));
                albedo = lerp(albedo, _RockColor.rgb, rock);

                float snow = smoothstep(_SnowHeight - _SnowBlend, _SnowHeight + _SnowBlend, h01 + (variation - 0.5) * 0.04);
                snow *= 1.0 - smoothstep(_RockSlope.y, _RockSlope.y + 0.25, slope); // cliffs stay bare
                albedo = lerp(albedo, _SnowColor.rgb, snow);

                // Wet: darker near water and under it.
                float wet = saturate(masks.g * 0.6 + smoothstep(-0.5, 0.2, waterDepth));
                albedo *= 1.0 - wet * _WetDarken;

                albedo *= (1.0 + (variation - 0.5) * 2.0 * _ColorNoise) * _Brightness;
                albedo *= saturate(1.0 + curvature * 0.6 * _Curvature); // creases darker, crests lighter

                // positionWS overload = URP fades shadows out at the shadow distance.
                Light light = GetMainLight(TransformWorldToShadowCoord(IN.positionWS), IN.positionWS, half4(1, 1, 1, 1));
                half ndl = saturate(dot(n, light.direction));
                half occlusion = saturate(1.0 - max(curvature, 0.0) * 0.5 * _Curvature);
                half3 color = albedo * (light.color * ndl * light.shadowAttenuation + SampleSH(n) * _Ambient * occlusion);
                color = MixFog(color, IN.fog);
                return half4(color, 1);
            }
            ENDHLSL
        }

        Pass
        {
            Name "ShadowCaster"
            Tags { "LightMode"="ShadowCaster" }
            ZWrite On ZTest LEqual ColorMask 0

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            float3 _LightDirection;
            float3 _LightPosition;

            struct Attributes { float4 positionOS : POSITION; UNITY_VERTEX_INPUT_INSTANCE_ID };

            float4 vert(Attributes IN) : SV_POSITION
            {
                UNITY_SETUP_INSTANCE_ID(IN);
                float3 posWS = GroundWorldPos(IN.positionOS.xyz);
                float3 n = GroundNormal(posWS.xz);
                #if _CASTING_PUNCTUAL_LIGHT_SHADOW
                    float3 lightDir = normalize(_LightPosition - posWS);
                #else
                    float3 lightDir = _LightDirection;
                #endif
                float4 cs = TransformWorldToHClip(ApplyShadowBias(posWS, n, lightDir));
                #if UNITY_REVERSED_Z
                    cs.z = min(cs.z, cs.w * UNITY_NEAR_CLIP_VALUE);
                #else
                    cs.z = max(cs.z, cs.w * UNITY_NEAR_CLIP_VALUE);
                #endif
                return cs;
            }
            half4 frag() : SV_Target { return 0; }
            ENDHLSL
        }

        Pass
        {
            Name "DepthOnly"
            Tags { "LightMode"="DepthOnly" }
            ZWrite On ColorMask R

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            struct Attributes { float4 positionOS : POSITION; UNITY_VERTEX_INPUT_INSTANCE_ID };
            float4 vert(Attributes IN) : SV_POSITION
            {
                UNITY_SETUP_INSTANCE_ID(IN);
                return TransformWorldToHClip(GroundWorldPos(IN.positionOS.xyz));
            }
            half4 frag() : SV_Target { return 0; }
            ENDHLSL
        }

        Pass
        {
            Name "DepthNormals"
            Tags { "LightMode"="DepthNormals" }
            ZWrite On

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile_instancing
            struct Attributes { float4 positionOS : POSITION; UNITY_VERTEX_INPUT_INSTANCE_ID };
            struct Varyings { float4 positionCS : SV_POSITION; float3 positionWS : TEXCOORD0; };
            Varyings vert(Attributes IN)
            {
                UNITY_SETUP_INSTANCE_ID(IN);
                Varyings OUT;
                OUT.positionWS = GroundWorldPos(IN.positionOS.xyz);
                OUT.positionCS = TransformWorldToHClip(OUT.positionWS);
                return OUT;
            }
            half4 frag(Varyings IN) : SV_Target
            {
                return half4(NormalizeNormalPerPixel(GroundNormal(IN.positionWS.xz)), 0);
            }
            ENDHLSL
        }
    }
}
