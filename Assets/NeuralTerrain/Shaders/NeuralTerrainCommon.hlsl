#ifndef NEURAL_TERRAIN_COMMON_INCLUDED
#define NEURAL_TERRAIN_COMMON_INCLUDED

// Shared by NeuralTerrain.shader (ground) and NeuralWater.shader (water).
// Everything here is set per draw from NeuralTerrain.cs (MaterialPropertyBlock).

#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

TEXTURE2D(_HeightTex);    SAMPLER(sampler_HeightTex);     // ground (river beds carved), 0..1 of max height
TEXTURE2D(_WaterTex);     SAMPLER(sampler_WaterTex);      // water surface, 0..1 of max height (below ground = dry)
TEXTURE2D(_MoistureTex);  SAMPLER(sampler_MoistureTex);   // r = moisture, g = shore/wet, b = ridge mask
TEXTURE2D(_BiomeLut);     SAMPLER(sampler_BiomeLut);      // x = height, y = moisture

float4 _TerrainParams;   // originX, originZ, worldSize, maxHeight
float4 _LodParams;       // range0, morphStart, leafNodeSize, gridResolution
float3 _LodViewerPos;
float  _HeightTexRes;
float  _TerrainBaseY;

float2 HeightUV(float2 worldXZ)
{
    float2 uv01 = (worldXZ - _TerrainParams.xy) / _TerrainParams.z;
    return uv01 * ((_HeightTexRes - 1.0) / _HeightTexRes) + 0.5 / _HeightTexRes; // texel centers
}

// UV into the 256x256 mask textures (moisture/shore/ridge), texel centers like HeightUV.
float2 MaskUV(float2 worldXZ)
{
    return (worldXZ - _TerrainParams.xy) / _TerrainParams.z * (255.0 / 256.0) + 0.5 / 256.0;
}

float SampleGround(float2 worldXZ)
{
    return SAMPLE_TEXTURE2D_LOD(_HeightTex, sampler_HeightTex, HeightUV(worldXZ), 0).r * _TerrainParams.w;
}

// Mip-filtered ground height (for shading only): far away / seen from above, one
// screen pixel covers many height samples, so read a blurrier mip instead of aliasing.
float SampleGroundLod(float2 worldXZ, float lod)
{
    return SAMPLE_TEXTURE2D_LOD(_HeightTex, sampler_HeightTex, HeightUV(worldXZ), lod).r * _TerrainParams.w;
}

float SampleWater(float2 worldXZ)
{
    return SAMPLE_TEXTURE2D_LOD(_WaterTex, sampler_WaterTex, HeightUV(worldXZ), 0).r * _TerrainParams.w;
}

float SampleSpacing() { return _TerrainParams.z / (_HeightTexRes - 1.0); }

float3 GroundNormal(float2 xz)
{
    float e = SampleSpacing();
    float hL = SampleGround(xz - float2(e, 0)), hR = SampleGround(xz + float2(e, 0));
    float hD = SampleGround(xz - float2(0, e)), hU = SampleGround(xz + float2(0, e));
    return normalize(float3(hL - hR, 2.0 * e, hD - hU));
}

// CDLOD: grid point (0..1 in the node) -> morphed world XZ.
// Vertices near the edge of this LOD's range slide onto the next, coarser
// grid, so neighbouring LODs meet exactly (no cracks) and nothing pops.
float2 CdlodWorldXZ(float3 positionOS)
{
    float4x4 m = GetObjectToWorldMatrix();
    float size = m._m00;
    float2 nodeMin = float2(m._m03, m._m23);
    float gridRes = _LodParams.w;

    float level = round(log2(size / _LodParams.z));
    float range = _LodParams.x * exp2(level);
    float prev = level > 0.5 ? range * 0.5 : 0.0;
    float morphStart = lerp(prev, range, _LodParams.y);
    float morphEnd = lerp(prev, range, 0.95);

    float2 gridPos = positionOS.xz;
    float dist = distance(nodeMin + gridPos * size, _LodViewerPos.xz);
    float k = saturate((dist - morphStart) / (morphEnd - morphStart));

    float2 fracPart = frac(gridPos * gridRes * 0.5) * 2.0 / gridRes;
    gridPos -= fracPart * k;
    return nodeMin + gridPos * size;
}

// Cheap procedural noise (no textures needed).
float NT_Hash(float2 p) { return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453); }
float NT_ValueNoise(float2 p)
{
    float2 i = floor(p), f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(lerp(NT_Hash(i), NT_Hash(i + float2(1, 0)), f.x),
                lerp(NT_Hash(i + float2(0, 1)), NT_Hash(i + float2(1, 1)), f.x), f.y);
}

#endif
