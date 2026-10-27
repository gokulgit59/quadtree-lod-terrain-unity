using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

/// <summary>
/// One CDLOD terrain (Continuous Distance-Dependent Level of Detail) with
/// rivers and lakes.
///
/// How it works, short version:
///  1. Heights are baked ONCE into a texture: the neural heightmap (256x256,
///     from a sketch) upsampled smoothly + fine detail noise. The sketch's blue
///     lines carve river channels, red lines raise and sharpen ridges.
///  2. A second texture holds the water surface (rivers + lake level).
///  3. Every frame a quadtree picks which squares to draw: small, dense squares
///     near the viewer, big coarse squares far away.
///  4. ALL squares share ONE grid mesh. The shaders move its vertices up to the
///     height (or water) texture and "morph" vertices near a LOD border so
///     there is no popping and no cracks.
///  5. Everything is drawn with GPU instancing: ~4 draw calls for the ground,
///     ~4 for water (only squares that contain water), no meshes rebuilt
///     while you move, no garbage.
///  6. Physics uses one TerrainCollider built from the same heights (play mode).
///
/// Works in edit mode too: change a value or press Generate in
/// Window > Neural Terrain Sketch and the terrain updates in the Scene view.
/// </summary>
[ExecuteAlways]
public class NeuralTerrain : MonoBehaviour
{
    public enum HeightResolution { Low_1025 = 1025, High_2049 = 2049 }

    const int S = 256; // sketch / model resolution

    [Header("Sketch result (set by the Sketch window)")]
    [Tooltip("256x256 heightmap from the neural model. Empty = noise terrain.")]
    public Texture2D sourceHeightmap;
    [Tooltip("The sketch it came from. Blue lines become rivers, red lines sharpen ridges.")]
    public Texture2D sourceSketch;

    [Header("Model output cleanup")]
    [Tooltip("Stretch the model output to use the full 0..1 height range (like matplotlib does when showing it). The raw output is very low-contrast.")]
    public bool autoContrast = true;
    [Range(0f, 5f), Tooltip("Ignore this % of extreme pixels at each end when stretching, so single spikes don't flatten everything else.")]
    public float contrastClipPercent = 0.3f;
    [Tooltip("Remove single-pixel spikes the model makes at green dots (needle-thin peaks).")]
    public bool despike = true;
    [Range(0, 4), Tooltip("Blur passes on the 256px output. Removes the small checkerboard/dot artifacts pix2pix leaves on drawn lines.")]
    public int smoothing = 2;

    [Header("Shape")]
    [Tooltip("Height of a pure-white heightmap pixel, in meters.")]
    public float maxHeight = 600f;
    [Tooltip("Remaps heights: x = input height 0..1, y = output 0..1. Bend it to flatten valleys or sharpen peaks.")]
    public AnimationCurve heightCurve = AnimationCurve.Linear(0, 0, 1, 1);
    [Tooltip("Height texture size. 2049 = 2 m per sample over 4 km. Use 1025 on very low-end devices.")]
    public HeightResolution resolution = HeightResolution.High_2049;

    [Header("Detail noise (adds small bumps the 256px heightmap can't hold)")]
    public float detailAmplitude = 10f;
    [Tooltip("Size of the biggest detail bumps, in meters.")]
    public float detailScale = 140f;
    [Range(1, 8)] public int detailOctaves = 4;
    [Range(0f, 1f), Tooltip("0 = smooth rolling ground (clean), 1 = sharp creases everywhere. Ridge lines always get sharp detail.")]
    public float detailRidges = 0.1f;
    [Range(0f, 1f), Tooltip("Bends the detail noise a little so it looks less regular. 0 = plain smooth noise.")]
    public float detailWarp = 0.3f;
    [Range(0f, 1f), Tooltip("How much more detail high ground gets than valleys.")]
    public float detailOnHighGround = 0.8f;
    public int seed = 1337;

    [Header("Noise terrain (only used with no sketch)")]
    public float fallbackScale = 1600f;

    [Header("Rivers (blue lines)")]
    public bool carveRivers = true;
    [Tooltip("How deep river channels are cut, in meters.")]
    public float riverDepth = 7f;
    [Range(1, 6), Tooltip("River channel width, in sketch pixels (16 m each).")]
    public int riverWidth = 2;

    [Header("Ridges (red lines)")]
    [Tooltip("Extra height added along ridge lines, in meters.")]
    public float ridgeHeight = 25f;
    [Range(1, 6), Tooltip("Ridge width, in sketch pixels (16 m each).")]
    public int ridgeWidth = 3;
    [Range(0f, 1f), Tooltip("Makes detail noise sharp and craggy along ridge lines.")]
    public float ridgeSharpen = 0.7f;

    [Header("Water")]
    [Tooltip("Material using Custom/NeuralWater. Empty = no water drawn.")]
    public Material waterMaterial;
    [Range(0f, 0.5f), Tooltip("Lake / sea level as a fraction of Max Height. Low basins below it fill with water. 0 = no lakes.")]
    public float lakeLevel = 0.05f;
    [Tooltip("Keep the Scene view redrawing (~30 fps) so water moves while you edit. Play mode always animates.")]
    public bool animateWaterInEditMode = true;

    [Header("Moisture (drives biomes)")]
    [Range(0f, 1f)] public float riverMoisture = 0.8f;
    [Tooltip("How far (sketch pixels, 16 m each) river wetness spreads.")]
    [Range(1, 24)] public int riverSpread = 8;
    [Range(0f, 1f)] public float noiseMoisture = 0.35f;
    public float moistureNoiseScale = 900f;

    [Header("Biomes (height x moisture)")]
    public List<BiomeDefinition> biomes = new List<BiomeDefinition>();
    [Range(0.01f, 0.2f)] public float biomeBlend = 0.06f;

    [Header("LOD (CDLOD)")]
    [Tooltip("Size of the smallest, most detailed square, in meters.")]
    public int leafNodeSize = 64;
    [Tooltip("Number of LOD levels. World size = leafNodeSize * 2^(levels-1). 64 m and 7 levels = 4096 m.")]
    [Range(2, 10)] public int lodLevels = 7;
    [Tooltip("Grid quads per square side. 32 = 2 m vertices up close. Use 16 on low-end devices.")]
    [Range(8, 64)] public int gridResolution = 32;
    [Tooltip("Detail distance. Level-0 range = leafNodeSize * this. Higher = more detail further out (more triangles).")]
    [Range(2.2f, 8f)] public float lodDistance = 3f;
    [Range(0.5f, 0.9f)] public float morphStart = 0.66f;
    [Tooltip("What the detail follows in play mode. Empty = Main Camera. In edit mode it follows the camera you're looking through.")]
    public Transform viewer;

    [Header("Rendering")]
    public Material material;
    public bool castShadows = true;

    [Header("Physics")]
    public bool generateCollider = true;

    // ---- runtime state ----
    Mesh _grid;
    Texture2D _heightTex, _waterTex, _maskTex, _biomeLut;
    float[] _heights, _water;    // res*res, 0..1 of maxHeight, row 0 = south (-Z)
    float[] _riverMask, _ridgeMask; // 256*256, 0..1
    bool[][] _waterCells;           // per LOD level: does this node contain water?
    int _res;
    TerrainCollider _collider;
    MaterialPropertyBlock _mpb;
    bool _dirty = true;

    readonly List<Matrix4x4>[] _quadrants = { new(), new(), new(), new() };
    readonly List<Matrix4x4>[] _waterQuadrants = { new(), new(), new(), new() };
    readonly Matrix4x4[] _batch = new Matrix4x4[1023];
    readonly Plane[] _planes = new Plane[6];
    float[] _ranges;

    static readonly int HeightTexId = Shader.PropertyToID("_HeightTex");
    static readonly int WaterTexId = Shader.PropertyToID("_WaterTex");
    static readonly int MoistureTexId = Shader.PropertyToID("_MoistureTex");
    static readonly int BiomeLutId = Shader.PropertyToID("_BiomeLut");
    static readonly int TerrainParamsId = Shader.PropertyToID("_TerrainParams");
    static readonly int LodParamsId = Shader.PropertyToID("_LodParams");
    static readonly int ViewerPosId = Shader.PropertyToID("_LodViewerPos");
    static readonly int HeightTexResId = Shader.PropertyToID("_HeightTexRes");
    static readonly int BaseYId = Shader.PropertyToID("_TerrainBaseY");

    public float WorldSize => leafNodeSize * (1 << (lodLevels - 1));
    Vector3 Origin => transform.position - new Vector3(WorldSize * 0.5f, 0f, WorldSize * 0.5f); // south-west corner

    // ------------------------------------------------------------------ lifecycle

    void OnEnable()
    {
        _dirty = true;
        RenderPipelineManager.beginCameraRendering += OnBeginCamera;
#if UNITY_EDITOR
        UnityEditor.EditorApplication.update += EditorTick;
#endif
    }

    void OnDisable()
    {
        RenderPipelineManager.beginCameraRendering -= OnBeginCamera;
#if UNITY_EDITOR
        UnityEditor.EditorApplication.update -= EditorTick;
#endif
        DestroyRuntime(_grid); DestroyRuntime(_heightTex); DestroyRuntime(_waterTex); DestroyRuntime(_maskTex); DestroyRuntime(_biomeLut);
        _grid = null; _heightTex = _waterTex = _maskTex = _biomeLut = null;
        if (_collider != null) { DestroyRuntime(_collider.terrainData); DestroyRuntime(_collider.gameObject); _collider = null; }
    }

    void OnValidate()
    {
        gridResolution = Mathf.Max(8, gridResolution & ~1); // must be even (quadrants)
        _dirty = true;
    }

    void Update()
    {
        if (_dirty) Rebuild();
    }

#if UNITY_EDITOR
    double _lastRepaint;

    // Edit mode only redraws on changes, so water would look frozen. Nudge the Scene view at ~30 fps.
    void EditorTick()
    {
        if (Application.isPlaying || !animateWaterInEditMode || waterMaterial == null) return;
        double now = UnityEditor.EditorApplication.timeSinceStartup;
        if (now - _lastRepaint < 1.0 / 30.0) return;
        _lastRepaint = now;
        UnityEditor.SceneView.RepaintAll();
    }
#endif

    /// <summary>Re-bakes heights, water, moisture, biome colors and the collider. Call after changing the source textures.</summary>
    public void Rebuild()
    {
        _dirty = false;
        BuildGridMesh();
        BuildSketchMasks();
        BakeHeights();
        BakeMasks();
        BakeBiomeLut();
        BuildWaterCells();
        if (Application.isPlaying && generateCollider) BuildCollider();
#if UNITY_EDITOR
        UnityEditor.SceneView.RepaintAll();
#endif
    }

    /// <summary>World-space ground height at x,z (bilinear, same data as the collider). Returns transform.y outside the terrain.</summary>
    public float SampleHeight(float worldX, float worldZ)
    {
        if (_dirty || _heights == null) Rebuild();
        return SampleField(_heights, worldX, worldZ);
    }

    /// <summary>World-space water surface height at x,z, or -Infinity where there's no water.</summary>
    public float SampleWater(float worldX, float worldZ)
    {
        if (_dirty || _water == null) Rebuild();
        float w = SampleField(_water, worldX, worldZ), g = SampleField(_heights, worldX, worldZ);
        return w > g + 0.05f ? w : float.NegativeInfinity;
    }

    float SampleField(float[] field, float worldX, float worldZ)
    {
        Vector3 o = Origin;
        float fx = (worldX - o.x) / WorldSize * (_res - 1);
        float fz = (worldZ - o.z) / WorldSize * (_res - 1);
        if (fx < 0 || fz < 0 || fx > _res - 1 || fz > _res - 1) return o.y;
        int x0 = Mathf.Min((int)fx, _res - 2), z0 = Mathf.Min((int)fz, _res - 2);
        float tx = fx - x0, tz = fz - z0;
        float a = Mathf.Lerp(field[z0 * _res + x0], field[z0 * _res + x0 + 1], tx);
        float b = Mathf.Lerp(field[(z0 + 1) * _res + x0], field[(z0 + 1) * _res + x0 + 1], tx);
        return o.y + Mathf.Lerp(a, b, tz) * maxHeight;
    }

    // ------------------------------------------------------------------ baking

    void BuildGridMesh()
    {
        int n = gridResolution;
        if (_grid != null && _grid.vertexCount == (n + 1) * (n + 1)) return;
        DestroyRuntime(_grid);

        var verts = new Vector3[(n + 1) * (n + 1)];
        for (int z = 0; z <= n; z++)
        for (int x = 0; x <= n; x++)
            verts[z * (n + 1) + x] = new Vector3(x / (float)n, 0f, z / (float)n);

        // 4 submeshes = 4 quadrants, so a node can draw just part of itself (CDLOD).
        int h = n / 2;
        var quads = new List<int>[4];
        for (int q = 0; q < 4; q++) quads[q] = new List<int>(h * h * 6);
        for (int z = 0; z < n; z++)
        for (int x = 0; x < n; x++)
        {
            int q = (x < h ? 0 : 1) + (z < h ? 0 : 2);
            int i = z * (n + 1) + x;
            quads[q].AddRange(new[] { i, i + n + 1, i + 1, i + 1, i + n + 1, i + n + 2 });
        }

        _grid = new Mesh { name = "CDLOD Grid", hideFlags = HideFlags.DontSave, vertices = verts, subMeshCount = 4 };
        for (int q = 0; q < 4; q++) _grid.SetTriangles(quads[q], q, false);
        _grid.bounds = new Bounds(new Vector3(0.5f, 0.5f, 0.5f), Vector3.one);
    }

    /// <summary>River (blue) and ridge (red) masks from the sketch, softened so channels and crests have smooth sides.</summary>
    void BuildSketchMasks()
    {
        _riverMask = new float[S * S];
        _ridgeMask = new float[S * S];
        if (sourceSketch == null || sourceSketch.width != S || sourceSketch.height != S || !sourceSketch.isReadable) return;

        Color32[] px = sourceSketch.GetPixels32();
        for (int i = 0; i < px.Length; i++)
        {
            Color32 c = px[i];
            if (c.b > 128 && c.r < 128 && c.g < 128) _riverMask[i] = 1f;
            else if (c.r > 128 && c.g < 128 && c.b < 128) _ridgeMask[i] = 1f;
        }
        SoftLine(_riverMask, riverWidth);
        SoftLine(_ridgeMask, ridgeWidth);
    }

    // Two box-blur passes, rescaled so a 1 px line is 1.0 at its center and fades over ~2*radius pixels.
    static void SoftLine(float[] m, int radius)
    {
        for (int pass = 0; pass < 2; pass++) { BoxBlur(m, radius, true); BoxBlur(m, radius, false); }
        float k = 2 * radius + 1;
        for (int i = 0; i < m.Length; i++) m[i] = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(m[i] * k));
    }

    void BakeHeights()
    {
        _res = (int)resolution;
        int count = _res * _res;
        float mh = Mathf.Max(1f, maxHeight);

        var source = new NativeArray<float>(sourceHeightmap != null ? S * S : 1, Allocator.TempJob);
        if (sourceHeightmap != null)
        {
            ReadHeightmap(sourceHeightmap, source);
            CleanupSource(source);
        }
        var river = new NativeArray<float>(_riverMask, Allocator.TempJob);
        var ridge = new NativeArray<float>(_ridgeMask, Allocator.TempJob);

        var curve = new NativeArray<float>(256, Allocator.TempJob);
        for (int i = 0; i < 256; i++) curve[i] = Mathf.Clamp01(heightCurve.Evaluate(i / 255f));

        bool r16 = SystemInfo.IsFormatSupported(GraphicsFormat.R16_UNorm, GraphicsFormatUsage.Sample); // half the memory of RFloat
        var heights = new NativeArray<float>(count, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var water = new NativeArray<float>(count, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var heights16 = new NativeArray<ushort>(r16 ? count : 0, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        var water16 = new NativeArray<ushort>(r16 ? count : 0, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
        new HeightBakeJob
        {
            source = source, hasSource = sourceHeightmap != null, river = river, ridge = ridge, curve = curve,
            heights = heights, water = water, heights16 = heights16, water16 = water16,
            res = _res, worldSize = WorldSize, maxHeight = mh,
            detailAmplitude = detailAmplitude, detailFrequency = 1f / Mathf.Max(1f, detailScale),
            detailOctaves = detailOctaves, ridges = detailRidges, highGround = detailOnHighGround,
            fallbackFrequency = 1f / Mathf.Max(1f, fallbackScale),
            offset = new float2((seed % 997) * 13.17f, (seed % 991) * 7.31f),
            riverDepth = carveRivers ? riverDepth / mh : 0f,
            ridgeHeight = ridgeHeight / mh, ridgeSharpen = ridgeSharpen,
            lakeLevel = lakeLevel, dryDrop = 0.5f / mh, warp = detailWarp,
        }.Schedule(_res, 16).Complete();

        if (_heights == null || _heights.Length != count) { _heights = new float[count]; _water = new float[count]; }
        heights.CopyTo(_heights);
        water.CopyTo(_water);
        if (r16)
        {
            Upload(ref _heightTex, GraphicsFormat.R16_UNorm, heights, heights16, "CDLOD Heights");
            Upload(ref _waterTex, GraphicsFormat.R16_UNorm, water, water16, "CDLOD Water");
        }
        else
        {
            Upload(ref _heightTex, GraphicsFormat.R32_SFloat, heights, heights16, "CDLOD Heights");
            Upload(ref _waterTex, GraphicsFormat.R32_SFloat, water, water16, "CDLOD Water");
        }

        source.Dispose(); river.Dispose(); ridge.Dispose(); curve.Dispose();
        heights.Dispose(); water.Dispose(); heights16.Dispose(); water16.Dispose();
    }

    /// <summary>
    /// Uploads a height field with a full mip chain built here in float precision.
    /// (Unity's own mip generation for 16-bit textures goes through 8 bits, which
    /// turns distant slopes into 2 m terraces / contour lines.)
    /// </summary>
    void Upload(ref Texture2D tex, GraphicsFormat format, NativeArray<float> level0, NativeArray<ushort> level0R16, string texName)
    {
        if (tex == null || tex.width != _res || tex.graphicsFormat != format)
        {
            DestroyRuntime(tex);
            // Mip chain: the fragment shader reads blurrier mips far away, so distant ground doesn't sparkle.
            tex = new Texture2D(_res, _res, format, TextureCreationFlags.MipChain)
                { name = texName, hideFlags = HideFlags.DontSave, wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Trilinear };
        }
        bool r16 = format == GraphicsFormat.R16_UNorm;
        if (r16) tex.SetPixelData(level0R16, 0); else tex.SetPixelData(level0, 0);

        var src = level0;
        int srcW = _res;
        for (int m = 1; m < tex.mipmapCount; m++)
        {
            int w = Mathf.Max(1, _res >> m);
            var dst = new NativeArray<float>(w * w, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var dst16 = new NativeArray<ushort>(r16 ? w * w : 0, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            new MipJob { src = src, srcW = srcW, dst = dst, dst16 = dst16, dstW = w }.Schedule(w, 16).Complete();
            if (r16) tex.SetPixelData(dst16, m); else tex.SetPixelData(dst, m);
            if (m > 1) src.Dispose();
            dst16.Dispose();
            src = dst;
            srcW = w;
        }
        if (tex.mipmapCount > 1) src.Dispose();
        tex.Apply(false, false);
    }

    [BurstCompile]
    struct MipJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> src;
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float> dst;
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<ushort> dst16;
        public int srcW, dstW;

        public void Execute(int y)
        {
            int y0 = math.min(2 * y, srcW - 1), y1 = math.min(2 * y + 1, srcW - 1);
            for (int x = 0; x < dstW; x++)
            {
                int x0 = math.min(2 * x, srcW - 1), x1 = math.min(2 * x + 1, srcW - 1);
                float v = (src[y0 * srcW + x0] + src[y0 * srcW + x1] + src[y1 * srcW + x0] + src[y1 * srcW + x1]) * 0.25f;
                dst[y * dstW + x] = v;
                if (dst16.Length > 0) dst16[y * dstW + x] = (ushort)(math.saturate(v) * 65535f + 0.5f);
            }
        }
    }

    void CleanupSource(NativeArray<float> src)
    {
        var a = src.ToArray();
        // 3x3 median: removes the 1-2 px spikes/pits pix2pix puts on green dots (needle peaks) without blurring shapes.
        if (despike)
        {
            var med = new float[a.Length];
            var win = new float[9];
            for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++)
            {
                int n = 0;
                for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++)
                    win[n++] = a[Mathf.Clamp(y + j, 0, S - 1) * S + Mathf.Clamp(x + i, 0, S - 1)];
                System.Array.Sort(win);
                med[y * S + x] = win[4];
            }
            a = med;
        }
        // 3x3 binomial blur passes: kills the 1-2 px pix2pix artifacts, keeps the shapes.
        for (int pass = 0; pass < smoothing; pass++)
        {
            var b = new float[a.Length];
            for (int y = 0; y < S; y++)
            for (int x = 0; x < S; x++)
            {
                float sum = 0f, w = 0f;
                for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++)
                {
                    int xx = Mathf.Clamp(x + i, 0, S - 1), yy = Mathf.Clamp(y + j, 0, S - 1);
                    float k = (i == 0 ? 2f : 1f) * (j == 0 ? 2f : 1f);
                    sum += a[yy * S + xx] * k; w += k;
                }
                b[y * S + x] = sum / w;
            }
            a = b;
        }
        if (autoContrast)
        {
            var sorted = (float[])a.Clone();
            System.Array.Sort(sorted);
            int clip = Mathf.Clamp((int)(sorted.Length * contrastClipPercent / 100f), 0, sorted.Length / 2 - 1);
            float lo = sorted[clip], hi = sorted[sorted.Length - 1 - clip];
            float inv = hi - lo > 1e-5f ? 1f / (hi - lo) : 1f;
            for (int i = 0; i < a.Length; i++) a[i] = Mathf.Clamp01((a[i] - lo) * inv);
        }
        src.CopyFrom(a);
    }

    static void ReadHeightmap(Texture2D tex, NativeArray<float> dst)
    {
        if (tex.width != S || tex.height != S) { Debug.LogWarning("[NeuralTerrain] sourceHeightmap must be 256x256", tex); return; }
        if (tex.format == TextureFormat.RFloat) { dst.CopyFrom(tex.GetPixelData<float>(0)); return; }
        var px = tex.GetPixels();
        for (int i = 0; i < dst.Length; i++) dst[i] = px[i].r;
    }

    [BurstCompile]
    struct HeightBakeJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float> source;
        [ReadOnly] public NativeArray<float> river;
        [ReadOnly] public NativeArray<float> ridge;
        [ReadOnly] public NativeArray<float> curve;
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float> heights;
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<float> water;
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<ushort> heights16; // GPU copies (R16), empty if unused
        [WriteOnly, NativeDisableParallelForRestriction] public NativeArray<ushort> water16;
        public bool hasSource;
        public int res, detailOctaves;
        public float worldSize, maxHeight, detailAmplitude, detailFrequency, ridges, highGround, fallbackFrequency;
        public float riverDepth, ridgeHeight, ridgeSharpen, lakeLevel, dryDrop, warp;
        public float2 offset;

        public void Execute(int z)
        {
            for (int x = 0; x < res; x++)
            {
                float2 uv = new float2(x, z) / (res - 1);
                float2 world = uv * worldSize + offset;

                float h = hasSource ? Bicubic(0, uv) : Fbm(world * fallbackFrequency, 6, 0f, 0.5f) * 0.5f + 0.5f;
                h = math.saturate(h);
                float c = h * 255f;
                int ci = math.min((int)c, 254);
                float baseH = math.lerp(curve[ci], curve[ci + 1], c - ci);

                float r = math.saturate(Bicubic(1, uv));  // 1 at river center
                float rg = math.saturate(Bicubic(2, uv)); // 1 on ridge crest
                baseH += rg * ridgeHeight;

                // Detail: craggier on ridges, calm in river beds.
                float sharp = math.lerp(ridges, 1f, rg * ridgeSharpen);
                float detail = Fbm(world * detailFrequency, detailOctaves, sharp, warp);
                detail *= detailAmplitude / maxHeight * math.lerp(1f, 0.3f + 1.4f * baseH, highGround);
                detail *= (1f - r) * (1f + rg * ridgeSharpen);

                float ground = baseH + detail - r * riverDepth;

                // Water surface: rivers sit 30% of the carve depth below the old ground
                // (so they fill the middle of the channel), lakes fill everything below lakeLevel.
                float w = -1f;
                if (riverDepth > 0f && r > 0.3f) w = baseH - riverDepth * 0.3f;
                if (lakeLevel > 0f && ground < lakeLevel) w = math.max(w, lakeLevel);
                if (w <= ground) w = ground - dryDrop; // dry: just under the ground

                int i = z * res + x;
                ground = math.saturate(ground);
                w = math.saturate(w);
                heights[i] = ground;
                water[i] = w;
                if (heights16.Length > 0)
                {
                    heights16[i] = (ushort)(ground * 65535f + 0.5f);
                    water16[i] = (ushort)(w * 65535f + 0.5f);
                }
            }
        }

        float Tap(int field, int index) => field == 0 ? source[index] : field == 1 ? river[index] : ridge[index];

        // Catmull-Rom over a 256x256 field (0 = heightmap, 1 = river mask, 2 = ridge mask): smooth, no 16 m staircase.
        float Bicubic(int field, float2 uv)
        {
            int len = field == 0 ? source.Length : field == 1 ? river.Length : ridge.Length;
            if (len < S * S) return 0f;
            float2 p = uv * (S - 1);
            int2 i = (int2)math.floor(p);
            float2 t = p - i;
            float result = 0f;
            for (int j = -1; j <= 2; j++)
            {
                float row = 0f;
                for (int k = -1; k <= 2; k++)
                    row += Tap(field, math.clamp(i.y + j, 0, S - 1) * S + math.clamp(i.x + k, 0, S - 1)) * CatmullRom(t.x, k);
                result += row * CatmullRom(t.y, j);
            }
            return result;
        }

        static float CatmullRom(float t, int k)
        {
            float t2 = t * t, t3 = t2 * t;
            switch (k)
            {
                case -1: return 0.5f * (-t3 + 2f * t2 - t);
                case 0: return 0.5f * (3f * t3 - 5f * t2 + 2f);
                case 1: return 0.5f * (-3f * t3 + 4f * t2 + t);
                default: return 0.5f * (t3 - t2);
            }
        }

        // -1..1 fBm (smooth simplex/Perlin-style noise), optionally blended toward
        // ridged noise for sharp crests. The warp offsets every octave by the SAME
        // small amount (it bends features, it doesn't stretch fine octaves into swirls).
        static float Fbm(float2 p, int octaves, float ridged, float warpAmount)
        {
            float2 w = new float2(noise.snoise(p * 0.5f + 31.7f), noise.snoise(p * 0.5f - 11.3f)) * warpAmount;
            float sum = 0f, amp = 1f, norm = 0f;
            for (int o = 0; o < octaves; o++)
            {
                float n = noise.snoise(p + w);
                float r = 1f - 2f * math.abs(n); // ridged: sharp crests
                sum += math.lerp(n, r, ridged) * amp;
                norm += amp;
                amp *= 0.45f;
                p = p * 2.03f + 17.1f;
            }
            return sum / norm;
        }
    }

    /// <summary>
    /// 256x256 mask texture for the shaders:
    ///   R = moisture (wet along rivers + soft noise + wetter lowlands) -> biomes
    ///   G = shore (near water)  -> sand and wet ground
    ///   B = ridge lines         -> extra rock
    /// </summary>
    void BakeMasks()
    {
        var wetRivers = (float[])_riverMask.Clone();
        for (int pass = 0; pass < 2; pass++) { BoxBlur(wetRivers, riverSpread, true); BoxBlur(wetRivers, riverSpread, false); }

        var shore = new float[S * S];
        for (int z = 0; z < S; z++)
        for (int x = 0; x < S; x++)
        {
            int i = (z * (_res - 1) / (S - 1)) * _res + x * (_res - 1) / (S - 1);
            shore[z * S + x] = _water[i] > _heights[i] ? 1f : 0f;
        }
        BoxBlur(shore, 2, true); BoxBlur(shore, 2, false);

        var tex = new Color32[S * S];
        float freq = WorldSize / Mathf.Max(1f, moistureNoiseScale);
        for (int z = 0; z < S; z++)
        for (int x = 0; x < S; x++)
        {
            int i = z * S + x;
            float n = Mathf.PerlinNoise(x / (float)S * freq + seed * 0.37f, z / (float)S * freq + 51.3f);
            float h = _heights[(z * (_res - 1) / (S - 1)) * _res + x * (_res - 1) / (S - 1)];
            float river = Mathf.Clamp01(wetRivers[i] * 3f) * riverMoisture;
            float wet = 0.35f + (n - 0.5f) * 2f * noiseMoisture + (0.5f - h) * 0.3f;
            wet = Mathf.Clamp01(Mathf.Max(wet, river) + river * 0.2f + shore[i] * 0.3f);
            float s = Mathf.Clamp01(shore[i] * 2.5f);
            tex[i] = new Color32((byte)(wet * 255f), (byte)(s * 255f), (byte)(_ridgeMask[i] * 255f), 255);
        }

        if (_maskTex == null)
            _maskTex = new Texture2D(S, S, TextureFormat.RGBA32, false, true)
                { name = "Terrain Masks", hideFlags = HideFlags.DontSave, wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
        _maskTex.SetPixels32(tex);
        _maskTex.Apply();
    }

    static void BoxBlur(float[] a, int r, bool horizontal)
    {
        var tmp = new float[S];
        for (int line = 0; line < S; line++)
        {
            for (int i = 0; i < S; i++)
            {
                float sum = 0;
                for (int k = -r; k <= r; k++)
                {
                    int j = Mathf.Clamp(i + k, 0, S - 1);
                    sum += horizontal ? a[line * S + j] : a[j * S + line];
                }
                tmp[i] = sum / (2 * r + 1);
            }
            for (int i = 0; i < S; i++)
                if (horizontal) a[line * S + i] = tmp[i]; else a[i * S + line] = tmp[i];
        }
    }

    /// <summary>
    /// 64x64 color table: x = height, y = moisture. Every biome adds its color
    /// weighted by how well it fits, so borders blend smoothly and colors never
    /// pop when LOD changes (the shader looks them up per pixel).
    /// </summary>
    void BakeBiomeLut()
    {
        const int L = 64;
        var px = new Color[L * L];
        for (int y = 0; y < L; y++)
        for (int x = 0; x < L; x++)
        {
            float h = x / (L - 1f), m = y / (L - 1f);
            Color sum = Color.black; float wsum = 0f;
            Color nearest = new Color(0.35f, 0.5f, 0.25f); float best = float.MaxValue;
            foreach (var b in biomes)
            {
                if (b == null) continue;
                float w = Axis(h, b.heightRange) * Axis(m, b.rainfallRange);
                sum += b.grassColor * w; wsum += w;
                float d = Mathf.Abs(h - Mathf.Clamp(h, b.heightRange.x, b.heightRange.y)) +
                          Mathf.Abs(m - Mathf.Clamp(m, b.rainfallRange.x, b.rainfallRange.y));
                if (d < best) { best = d; nearest = b.grassColor; }
            }
            px[y * L + x] = wsum > 0.001f ? Color.Lerp(nearest, sum / wsum, Mathf.Clamp01(wsum * 4f)) : nearest;
        }

        if (_biomeLut == null)
            _biomeLut = new Texture2D(L, L, TextureFormat.RGBA32, false, false)
                { name = "Biome LUT", hideFlags = HideFlags.DontSave, wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
        _biomeLut.SetPixels(px);
        _biomeLut.Apply();
    }

    float Axis(float v, Vector2 range)
    {
        float m = biomeBlend;
        float rise = Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(range.x - m, range.x + m, v));
        float fall = 1f - Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(range.y - m, range.y + m, v));
        // Edges of the 0..1 table count as "inside" so a biome reaching 0 or 1 isn't faded there.
        if (range.x <= 0f) rise = 1f;
        if (range.y >= 1f) fall = 1f;
        return rise * fall;
    }

    /// <summary>Which quadtree nodes contain any water, per LOD level, so water is only drawn where it exists.</summary>
    void BuildWaterCells()
    {
        int cells = 1 << (lodLevels - 1); // leaf nodes per side
        _waterCells = new bool[lodLevels][];
        var leaf = new bool[cells * cells];
        float step = (_res - 1f) / cells;
        for (int z = 0; z < _res; z++)
        for (int x = 0; x < _res; x++)
        {
            int i = z * _res + x;
            if (_water[i] <= _heights[i]) continue;
            // A sample on a cell border belongs to both neighbours.
            int cx0 = Mathf.Clamp(Mathf.FloorToInt((x - 1) / step), 0, cells - 1), cx1 = Mathf.Clamp(Mathf.FloorToInt((x + 1) / step), 0, cells - 1);
            int cz0 = Mathf.Clamp(Mathf.FloorToInt((z - 1) / step), 0, cells - 1), cz1 = Mathf.Clamp(Mathf.FloorToInt((z + 1) / step), 0, cells - 1);
            for (int cz = cz0; cz <= cz1; cz++)
            for (int cx = cx0; cx <= cx1; cx++)
                leaf[cz * cells + cx] = true;
        }
        _waterCells[0] = leaf;
        for (int l = 1; l < lodLevels; l++)
        {
            int n = cells >> l, prevN = n * 2;
            var prev = _waterCells[l - 1];
            var cur = new bool[n * n];
            for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
                cur[z * n + x] = prev[(2 * z) * prevN + 2 * x] || prev[(2 * z) * prevN + 2 * x + 1] ||
                                 prev[(2 * z + 1) * prevN + 2 * x] || prev[(2 * z + 1) * prevN + 2 * x + 1];
            _waterCells[l] = cur;
        }
    }

    bool HasWater(Vector2 min, int level)
    {
        if (_waterCells == null || level >= _waterCells.Length) return false;
        int n = (1 << (lodLevels - 1)) >> level;
        float size = leafNodeSize * (1 << level);
        Vector3 o = Origin;
        int x = Mathf.Clamp(Mathf.RoundToInt((min.x - o.x) / size), 0, n - 1);
        int z = Mathf.Clamp(Mathf.RoundToInt((min.y - o.z) / size), 0, n - 1);
        return _waterCells[level][z * n + x];
    }

    void BuildCollider()
    {
        if (_collider == null)
        {
            var go = new GameObject("Terrain Collider (runtime)") { hideFlags = HideFlags.DontSave | HideFlags.NotEditable };
            go.transform.SetParent(transform, false);
            go.layer = gameObject.layer;
            _collider = go.AddComponent<TerrainCollider>();
        }
        var data = _collider.terrainData;
        if (data == null || data.heightmapResolution != _res)
        {
            DestroyRuntime(data);
            data = new TerrainData { heightmapResolution = _res };
        }
        data.size = new Vector3(WorldSize, maxHeight, WorldSize);
        var h2 = new float[_res, _res];
        for (int z = 0; z < _res; z++)
        for (int x = 0; x < _res; x++)
            h2[z, x] = _heights[z * _res + x];
        data.SetHeights(0, 0, h2);
        _collider.terrainData = data;
        _collider.transform.position = Origin;
    }

    // ------------------------------------------------------------------ CDLOD selection + drawing

    void OnBeginCamera(ScriptableRenderContext ctx, Camera cam)
    {
        if (cam.cameraType != CameraType.Game && cam.cameraType != CameraType.SceneView) return;
        if (material == null || _heightTex == null || _grid == null || _waterCells == null) return;

        Vector3 viewerPos = cam.transform.position;
        if (Application.isPlaying)
        {
            Transform v = viewer != null ? viewer : (Camera.main != null ? Camera.main.transform : null);
            if (v != null) viewerPos = v.position;
        }

        BuildRanges();
        GeometryUtility.CalculateFrustumPlanes(cam, _planes);
        foreach (var q in _quadrants) q.Clear();
        foreach (var q in _waterQuadrants) q.Clear();

        Vector3 o = Origin;
        float root = WorldSize;
        Select(new Vector2(o.x, o.z), root, lodLevels - 1, viewerPos, QualitySettings.shadowDistance);

        _mpb ??= new MaterialPropertyBlock();
        _mpb.SetTexture(HeightTexId, _heightTex);
        _mpb.SetTexture(WaterTexId, _waterTex);
        _mpb.SetTexture(MoistureTexId, _maskTex);
        _mpb.SetTexture(BiomeLutId, _biomeLut);
        _mpb.SetVector(TerrainParamsId, new Vector4(o.x, o.z, WorldSize, maxHeight));
        _mpb.SetVector(LodParamsId, new Vector4(_ranges[0], morphStart, leafNodeSize, gridResolution));
        _mpb.SetVector(ViewerPosId, viewerPos);
        _mpb.SetFloat(HeightTexResId, _res);
        _mpb.SetFloat(BaseYId, transform.position.y);

        var bounds = new Bounds(transform.position + Vector3.up * maxHeight * 0.5f, new Vector3(root, maxHeight + 100f, root));
        var rp = new RenderParams(material)
        {
            camera = cam,
            layer = gameObject.layer,
            matProps = _mpb,
            shadowCastingMode = castShadows ? ShadowCastingMode.On : ShadowCastingMode.Off,
            receiveShadows = true,
            lightProbeUsage = LightProbeUsage.Off, // "Off" still feeds the scene's ambient probe to SampleSH
            worldBounds = bounds,
        };
        Draw(rp, _quadrants);

        if (waterMaterial != null)
        {
            rp.material = waterMaterial;
            rp.shadowCastingMode = ShadowCastingMode.Off;
            Draw(rp, _waterQuadrants);
        }
    }

    void Draw(RenderParams rp, List<Matrix4x4>[] lists)
    {
        for (int q = 0; q < 4; q++)
        {
            var list = lists[q];
            for (int start = 0; start < list.Count; start += _batch.Length)
            {
                int n = Mathf.Min(_batch.Length, list.Count - start);
                list.CopyTo(start, _batch, 0, n);
                Graphics.RenderMeshInstanced(rp, _grid, q, _batch, n);
            }
        }
    }

    void BuildRanges()
    {
        if (_ranges == null || _ranges.Length != lodLevels) _ranges = new float[lodLevels];
        for (int l = 0; l < lodLevels; l++) _ranges[l] = leafNodeSize * lodDistance * (1 << l);
    }

    /// <summary>
    /// Strugar's CDLOD selection. Returns false if this node is beyond its
    /// level's range (the parent then draws that area itself, coarser).
    /// </summary>
    bool Select(Vector2 min, float size, int level, Vector3 viewerPos, float shadowDist)
    {
        if (!InRange(min, size, viewerPos, _ranges[level])) return false;

        var bounds = new Bounds(new Vector3(min.x + size * 0.5f, transform.position.y + maxHeight * 0.5f, min.y + size * 0.5f),
                                new Vector3(size, maxHeight + 20f, size));
        // Off-screen nodes are skipped (but kept close by, so they still cast shadows into view).
        if (!GeometryUtility.TestPlanesAABB(_planes, bounds) && !InRange(min, size, viewerPos, shadowDist))
            return true;

        if (level == 0 || !InRange(min, size, viewerPos, _ranges[level - 1]))
        {
            var m = NodeMatrix(min, size);
            bool water = HasWater(min, level);
            for (int q = 0; q < 4; q++)
            {
                _quadrants[q].Add(m);
                if (water) _waterQuadrants[q].Add(m);
            }
            return true;
        }

        float half = size * 0.5f;
        for (int q = 0; q < 4; q++)
        {
            var childMin = min + new Vector2((q & 1) * half, (q >> 1) * half);
            if (!Select(childMin, half, level - 1, viewerPos, shadowDist))
            {
                // Draw just this quarter at our coarser level.
                var m = NodeMatrix(min, size);
                _quadrants[q].Add(m);
                if (HasWater(childMin, level - 1)) _waterQuadrants[q].Add(m);
            }
        }
        return true;
    }

    Matrix4x4 NodeMatrix(Vector2 min, float size) =>
        Matrix4x4.TRS(new Vector3(min.x, transform.position.y, min.y), Quaternion.identity, new Vector3(size, 1f, size));

    static bool InRange(Vector2 min, float size, Vector3 p, float range)
    {
        float dx = Mathf.Max(min.x - p.x, 0f, p.x - (min.x + size));
        float dz = Mathf.Max(min.y - p.z, 0f, p.z - (min.y + size));
        return dx * dx + dz * dz < range * range;
    }

    static void DestroyRuntime(Object o)
    {
        if (o == null) return;
        if (Application.isPlaying) Destroy(o); else DestroyImmediate(o);
    }
}
