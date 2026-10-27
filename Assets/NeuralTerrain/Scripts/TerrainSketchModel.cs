using System;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// Day 5: runs the pix2pix generator. Sketch (256x256, black background,
/// red ridges, blue rivers, green peaks) in, heightmap out.
///
/// Model contract (from training):
///   input  "sketch"    float (1,3,256,256), RGB in -1..1, top row first
///   output "heightmap" same shape, -1..1, all 3 channels equal
///   height01 = (value + 1) / 2
///
/// Plain C# class so the editor Sketch window and play mode can both use it.
/// Dispose() it when done -- the Worker holds GPU memory.
/// </summary>
public sealed class TerrainSketchModel : IDisposable
{
    public const int Size = 256;
    const int Pixels = Size * Size;

    readonly Model _model;
    readonly BackendType _backend;
    Worker _worker;

    public TerrainSketchModel(ModelAsset asset, BackendType backend = BackendType.GPUCompute)
    {
        if (asset == null) throw new ArgumentNullException(nameof(asset));
        _model = ModelLoader.Load(asset);
        // Pixel-shader backend for devices without compute shaders (older GLES).
        _backend = backend == BackendType.GPUCompute && !SystemInfo.supportsComputeShaders
            ? BackendType.GPUPixel : backend;
        _worker = new Worker(_model, _backend);
    }

    /// <summary>
    /// Runs the model. Returns 256*256 heights in 0..1, row 0 = BOTTOM row
    /// (Unity texture order), ready for Texture2D.SetPixelData / terrain baking.
    /// </summary>
    public float[] Generate(Texture2D sketch) => Generate(sketch, out _);

    /// <param name="rawOutput">The raw (1,3,256,256) -1..1 output, top row first, for verification.</param>
    public float[] Generate(Texture2D sketch, out float[] rawOutput)
    {
        using var input = new Tensor<float>(new TensorShape(1, 3, Size, Size), SketchToTensorData(sketch));
        _worker.Schedule(input);
        var output = _worker.PeekOutput("heightmap") as Tensor<float>;
        rawOutput = output.DownloadToArray();

        var heights = new float[Pixels];
        for (int y = 0; y < Size; y++)
        {
            int srcRow = (Size - 1 - y) * Size; // model is top-first, Unity is bottom-first
            for (int x = 0; x < Size; x++)
            {
                // Channels are equal on average, but ~1% of pixels (mostly on
                // drawn lines) carry small GAN artifacts -- averaging cleans them.
                int i = srcRow + x;
                float v = (rawOutput[i] + rawOutput[Pixels + i] + rawOutput[2 * Pixels + i]) / 3f;
                heights[y * Size + x] = Mathf.Clamp01((v + 1f) * 0.5f);
            }
        }
        return heights;
    }

    /// <summary>
    /// Builds the NCHW input by hand from the raw bytes (no GPU conversion),
    /// so sRGB/linear texture settings can't silently change the values.
    /// </summary>
    public static float[] SketchToTensorData(Texture2D sketch)
    {
        if (sketch.width != Size || sketch.height != Size)
            throw new ArgumentException($"Sketch must be {Size}x{Size}, got {sketch.width}x{sketch.height}");

        Color32[] px = sketch.GetPixels32();
        var data = new float[3 * Pixels];
        for (int y = 0; y < Size; y++)
        {
            int dstRow = (Size - 1 - y) * Size; // flip to top row first
            for (int x = 0; x < Size; x++)
            {
                Color32 c = px[y * Size + x];
                int i = dstRow + x;
                data[i] = c.r / 127.5f - 1f;
                data[Pixels + i] = c.g / 127.5f - 1f;
                data[2 * Pixels + i] = c.b / 127.5f - 1f;
            }
        }
        return data;
    }

    /// <summary>
    /// "Known good output" check inside Unity: the CPU backend is the
    /// reference implementation (plain float math, same as PyTorch on CPU),
    /// so the GPU result must match it. Also checks the model's own contract:
    /// output in -1..1 and all three channels equal.
    /// </summary>
    public static string Verify(ModelAsset asset, Texture2D sketch, out bool passed)
    {
        float[] gpu, cpu;
        using (var g = new TerrainSketchModel(asset, BackendType.GPUCompute)) g.Generate(sketch, out gpu);
        using (var c = new TerrainSketchModel(asset, BackendType.CPU)) c.Generate(sketch, out cpu);

        double maxErr = 0, sumErr = 0, channelDiff = 0, mean = 0, sq = 0;
        float min = float.MaxValue, max = float.MinValue;
        for (int i = 0; i < gpu.Length; i++)
        {
            double e = Math.Abs(gpu[i] - cpu[i]);
            maxErr = Math.Max(maxErr, e);
            sumErr += e;
            min = Mathf.Min(min, cpu[i]);
            max = Mathf.Max(max, cpu[i]);
        }
        for (int i = 0; i < Pixels; i++)
        {
            channelDiff += Math.Max(Math.Abs(cpu[i] - cpu[Pixels + i]), Math.Abs(cpu[i] - cpu[2 * Pixels + i]));
            mean += cpu[i];
            sq += cpu[i] * cpu[i];
        }
        mean /= Pixels;
        channelDiff /= Pixels;
        double std = Math.Sqrt(Math.Max(0, sq / Pixels - mean * mean));

        bool matches = maxErr < 1e-2;
        bool inRange = min >= -1.001f && max <= 1.001f;
        bool channelsEqual = channelDiff < 0.02; // mean; a few line pixels differ more (GAN artifacts)
        bool notFlat = std > 0.01;
        passed = matches && inRange && channelsEqual && notFlat;

        return $"[Verify] {(passed ? "PASS" : "FAIL")}\n" +
               $"  GPU vs CPU reference: max error {maxErr:E2}, mean error {sumErr / gpu.Length:E2} {(matches ? "ok" : "TOO HIGH")}\n" +
               $"  Output range: {min:F3} .. {max:F3} {(inRange ? "ok" : "OUT OF -1..1")}\n" +
               $"  Channels equal: mean diff {channelDiff:E2} {(channelsEqual ? "ok" : "NOT EQUAL")}\n" +
               $"  Height variation (std): {std:F3} {(notFlat ? "ok" : "FLAT OUTPUT")}";
    }

    /// <summary>A fixed example sketch (seed 7), so results are comparable.</summary>
    public static Texture2D CreateTestSketch() => CreateRandomSketch(7);

    /// <summary>
    /// A random sketch in the training style: red ridge network, blue rivers,
    /// green peaks on black. Same seed = same sketch = same terrain.
    /// Dense networks work best -- the model responds to line density far more
    /// than to single strokes (sparse sketches come out flat).
    /// </summary>
    public static Texture2D CreateRandomSketch(int seed, int ridges = 12, int rivers = 10, int peaks = 10)
    {
        var tex = new Texture2D(Size, Size, TextureFormat.RGBA32, false, true) { name = $"Sketch_{seed}" };
        var px = new Color32[Pixels];
        for (int i = 0; i < px.Length; i++) px[i] = new Color32(0, 0, 0, 255);

        var red = new Color32(255, 0, 0, 255);
        var blue = new Color32(0, 0, 255, 255);
        var green = new Color32(0, 255, 0, 255);
        var rng = new System.Random(seed);
        var ridgePoints = new System.Collections.Generic.List<Vector2>();
        float Rand() => (float)rng.NextDouble();

        void Dot(Vector2 p, Color32 c, float r)
        {
            int ri = Mathf.CeilToInt(r);
            for (int y = -ri; y <= ri; y++)
            for (int x = -ri; x <= ri; x++)
            {
                int px0 = (int)p.x + x, py0 = (int)p.y + y;
                if (x * x + y * y <= r * r && px0 >= 0 && py0 >= 0 && px0 < Size && py0 < Size)
                    px[py0 * Size + px0] = c;
            }
        }
        void Line(Vector2 a, Vector2 b, Color32 c, float r)
        {
            int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b)));
            for (int s = 0; s <= steps; s++) Dot(Vector2.Lerp(a, b, s / (float)steps), c, r);
        }
        bool Inside(Vector2 p) => p.x >= 0 && p.y >= 0 && p.x < Size && p.y < Size;
        bool IsRidge(Vector2 p) => Inside(p) && px[(int)p.y * Size + (int)p.x].r > 128;

        // Ridges: wandering crests that sometimes branch into side spurs.
        void Ridge(Vector2 p, float angle, int length, int depth)
        {
            for (int s = 0; s < length; s++)
            {
                angle += (Rand() - 0.5f) * 0.45f;
                var next = p + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * 1.5f;
                if (!Inside(next)) return;
                Line(p, next, red, 0.7f);
                if (s % 6 == 0) ridgePoints.Add(next);
                if (depth < 2 && Rand() < 0.012f)
                    Ridge(next, angle + (Rand() < 0.5f ? 1f : -1f) * (0.6f + Rand() * 0.6f), length / 3, depth + 1);
                p = next;
            }
        }
        for (int i = 0; i < ridges; i++)
            Ridge(new Vector2(Rand() * Size, Rand() * Size), Rand() * Mathf.PI * 2f, 120 + rng.Next(120), 0);

        // Rivers: long, gently curving, steer away from ridges, often start/end at the map edge.
        for (int i = 0; i < rivers; i++)
        {
            Vector2 p;
            float angle;
            if (Rand() < 0.6f)
            {
                int edge = rng.Next(4);
                float t = Rand() * Size;
                p = edge switch { 0 => new Vector2(t, 0), 1 => new Vector2(Size - 1, t), 2 => new Vector2(t, Size - 1), _ => new Vector2(0, t) };
                angle = edge * Mathf.PI * 0.5f + Mathf.PI * 0.5f + (Rand() - 0.5f) * 0.8f;
            }
            else { p = new Vector2(Rand() * Size, Rand() * Size); angle = Rand() * Mathf.PI * 2f; }

            int length = 160 + rng.Next(160);
            for (int s = 0; s < length; s++)
            {
                angle += (Rand() - 0.5f) * 0.3f;
                var ahead = p + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * 4f;
                if (IsRidge(ahead)) angle += (Rand() < 0.5f ? 1f : -1f) * 0.8f; // rivers don't cross ridges
                var next = p + new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * 1.5f;
                if (!Inside(next)) break;
                if (IsRidge(next)) continue;
                Line(p, next, blue, 0.7f);
                p = next;
            }
        }

        // Peaks: mostly on ridges, a few anywhere.
        for (int i = 0; i < peaks; i++)
        {
            Vector2 p = ridgePoints.Count > 0 && Rand() < 0.8f
                ? ridgePoints[rng.Next(ridgePoints.Count)]
                : new Vector2(Rand() * Size, Rand() * Size);
            Dot(p, green, 1f);
        }

        tex.SetPixels32(px);
        tex.Apply();
        return tex;
    }

    public void Dispose()
    {
        _worker?.Dispose();
        _worker = null;
    }
}
