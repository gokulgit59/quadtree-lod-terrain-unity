using System.Collections.Generic;
using System.IO;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Window > Neural Terrain Sketch.
/// Paint a sketch, press Generate: the model turns it into a heightmap and the
/// NeuralTerrain in the open scene updates right away (edit mode or play mode).
///
///   Red line   = ridge / mountain crest
///   Blue line  = river / valley (also makes the land around it wetter)
///   Green dot  = peak
/// </summary>
public class NeuralTerrainSketchWindow : EditorWindow
{
    const int Size = TerrainSketchModel.Size;

    // Paths are found from this script's own location (<root>/Scripts/Editor/this.cs),
    // so the folder works wherever it's imported or moved to.
    string Root
    {
        get
        {
            string script = AssetDatabase.GetAssetPath(MonoScript.FromScriptableObject(this));
            return Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(script))).Replace('\\', '/');
        }
    }
    string ModelPath
    {
        get
        {
            string p = Root + "/Models/terrain_generator.onnx";
            if (AssetDatabase.LoadAssetAtPath<ModelAsset>(p) != null) return p;
            string[] found = AssetDatabase.FindAssets("terrain_generator t:ModelAsset");
            return found.Length > 0 ? AssetDatabase.GUIDToAssetPath(found[0]) : p;
        }
    }
    string OutputFolder => Root + "/Generated";

    enum Tool { Ridge, River, Peak, Erase }

    static readonly Color32 Black = new(0, 0, 0, 255);
    static readonly Color32[] ToolColors = { new(255, 0, 0, 255), new(0, 0, 255, 255), new(0, 255, 0, 255), new(0, 0, 0, 255) };
    static readonly string[] ToolNames = { "Ridge (red)", "River (blue)", "Peak (green dot)", "Erase" };

    NeuralTerrain _terrain;
    ModelAsset _modelAsset;
    TerrainSketchModel _model;

    Texture2D _canvas, _preview;
    Color32[] _pixels;
    readonly Stack<Color32[]> _undo = new();
    Tool _tool = Tool.Ridge;
    float _brush = 1.5f;
    bool _liveGenerate = true;
    Vector2? _lastPaint;
    string _status = "";
    Vector2 _scroll;

    // Random sketch settings (kept across domain reloads).
    [SerializeField] int _seed = 7;
    [SerializeField] int _ridges = 12;
    [SerializeField] int _rivers = 10;
    [SerializeField] int _peaks = 10;

    [MenuItem("Window/Neural Terrain Sketch")]
    static void Open() => GetWindow<NeuralTerrainSketchWindow>("Neural Terrain Sketch").minSize = new Vector2(560, 720);

    void OnEnable()
    {
        _modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);
        _terrain = FindAnyObjectByType<NeuralTerrain>();
        _canvas = new Texture2D(Size, Size, TextureFormat.RGBA32, false, true) { filterMode = FilterMode.Point, hideFlags = HideFlags.DontSave };
        _preview = new Texture2D(Size, Size, TextureFormat.RGBA32, false, true) { filterMode = FilterMode.Bilinear, hideFlags = HideFlags.DontSave };
        _pixels = new Color32[Size * Size];
        if (_terrain != null && _terrain.sourceSketch != null) LoadFrom(_terrain);
        else Clear(false);
        AssemblyReloadEvents.beforeAssemblyReload += DisposeModel;
    }

    void OnDisable()
    {
        AssemblyReloadEvents.beforeAssemblyReload -= DisposeModel;
        DisposeModel();
        DestroyImmediate(_canvas);
        DestroyImmediate(_preview);
    }

    void DisposeModel() { _model?.Dispose(); _model = null; }

    // ------------------------------------------------------------------ UI

    void OnGUI()
    {
        _scroll = EditorGUILayout.BeginScrollView(_scroll);

        // References are lost when scenes change or scripts recompile -- find them again.
        if (_terrain == null) _terrain = FindAnyObjectByType<NeuralTerrain>();
        if (_modelAsset == null) _modelAsset = AssetDatabase.LoadAssetAtPath<ModelAsset>(ModelPath);

        _terrain = (NeuralTerrain)EditorGUILayout.ObjectField("Terrain", _terrain, typeof(NeuralTerrain), true);
        var newModel = (ModelAsset)EditorGUILayout.ObjectField("Model", _modelAsset, typeof(ModelAsset), false);
        if (newModel != _modelAsset) { _modelAsset = newModel; DisposeModel(); }
        if (_terrain == null) EditorGUILayout.HelpBox("No NeuralTerrain in the open scene. Drag Prefabs/NeuralTerrain into the scene -- Generate turns on automatically.", MessageType.Warning);
        if (_modelAsset == null) EditorGUILayout.HelpBox($"Model not found at {ModelPath}. Drag terrain_generator.onnx into the Model field.", MessageType.Warning);

        EditorGUILayout.Space();
        _tool = (Tool)GUILayout.Toolbar((int)_tool, ToolNames);
        _brush = EditorGUILayout.Slider("Brush size", _brush, 0.5f, 6f);

        // Canvas (left) and generated heightmap (right).
        float side = Mathf.Min((position.width - 30f) * 0.5f, 512f);
        EditorGUILayout.BeginHorizontal();
        Rect canvasRect = GUILayoutUtility.GetRect(side, side, GUILayout.ExpandWidth(false));
        GUILayout.Space(8);
        Rect previewRect = GUILayoutUtility.GetRect(side, side, GUILayout.ExpandWidth(false));
        EditorGUILayout.EndHorizontal();
        GUI.DrawTexture(canvasRect, _canvas, ScaleMode.StretchToFill, false);
        GUI.DrawTexture(previewRect, _preview, ScaleMode.StretchToFill, false);
        EditorGUILayout.BeginHorizontal();
        GUILayout.Label("Sketch (paint here)", GUILayout.Width(side + 8));
        GUILayout.Label("Heightmap (model output)");
        EditorGUILayout.EndHorizontal();
        HandlePainting(canvasRect);

        EditorGUILayout.Space();
        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Undo", GUILayout.Height(24)) && _undo.Count > 0) { _pixels = _undo.Pop(); ApplyCanvas(); }
        if (GUILayout.Button("Clear", GUILayout.Height(24))) Clear(true);
        if (GUILayout.Button("Save copy", GUILayout.Height(24))) SaveCopy();
        if (GUILayout.Button("Load sketch...", GUILayout.Height(24))) LoadSketchFile();
        EditorGUILayout.EndHorizontal();

        // Random terrain: fills the canvas (undoable), generates; then keep painting on top of it.
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Random terrain", EditorStyles.boldLabel);
        EditorGUILayout.BeginHorizontal();
        _seed = EditorGUILayout.IntField("Seed", _seed);
        if (GUILayout.Button("Use seed", GUILayout.Width(90))) RandomSketch(_seed);
        if (GUILayout.Button("New random", GUILayout.Width(110))) RandomSketch(_seed = UnityEngine.Random.Range(1, 100000));
        EditorGUILayout.EndHorizontal();
        _ridges = EditorGUILayout.IntSlider("Ridges", _ridges, 0, 30);
        _rivers = EditorGUILayout.IntSlider("Rivers", _rivers, 0, 30);
        _peaks = EditorGUILayout.IntSlider("Peaks", _peaks, 0, 40);

        _liveGenerate = EditorGUILayout.ToggleLeft("Generate automatically after every stroke", _liveGenerate);
        GUI.enabled = _terrain != null && _modelAsset != null;
        if (GUILayout.Button("Generate", GUILayout.Height(36))) Generate();
        GUI.enabled = _modelAsset != null;
        if (GUILayout.Button("Verify model (GPU vs CPU reference)")) RunVerify();
        GUI.enabled = true;

        if (!string.IsNullOrEmpty(_status)) EditorGUILayout.HelpBox(_status, MessageType.Info);
        EditorGUILayout.HelpBox("Red = ridges, Blue = rivers (carved + filled with water), Green dot = peaks. Black = lowland.\n" +
                                "Dense networks of lines work best; a few lone strokes come out flat.\n" +
                                "Terrain shape, detail, LOD and biomes are on the NeuralTerrain component; colors are on its material.", MessageType.None);
        EditorGUILayout.EndScrollView();
    }

    void HandlePainting(Rect rect)
    {
        Event e = Event.current;
        if (!rect.Contains(e.mousePosition) && e.type != EventType.MouseUp) return;

        Vector2 p = new Vector2((e.mousePosition.x - rect.x) / rect.width * Size,
                                (1f - (e.mousePosition.y - rect.y) / rect.height) * Size); // GUI is y-down, textures y-up

        if (e.type == EventType.MouseDown && e.button == 0)
        {
            PushUndo();
            if (_tool == Tool.Peak) Dot(p, ToolColors[(int)Tool.Peak], _brush + 1.5f);
            else Dot(p, ToolColors[(int)_tool], _brush);
            _lastPaint = p;
            ApplyCanvas(); e.Use();
        }
        else if (e.type == EventType.MouseDrag && e.button == 0 && _lastPaint.HasValue && _tool != Tool.Peak)
        {
            Line(_lastPaint.Value, p, ToolColors[(int)_tool], _brush);
            _lastPaint = p;
            ApplyCanvas(); e.Use();
        }
        else if (e.type == EventType.MouseUp && _lastPaint.HasValue)
        {
            _lastPaint = null;
            if (_liveGenerate && _terrain != null && _modelAsset != null) Generate();
            e.Use();
        }
    }

    // ------------------------------------------------------------------ painting

    void Line(Vector2 a, Vector2 b, Color32 c, float r)
    {
        int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(a, b) * 2f));
        for (int s = 0; s <= steps; s++) Dot(Vector2.Lerp(a, b, s / (float)steps), c, r);
    }

    void Dot(Vector2 p, Color32 c, float r)
    {
        int ri = Mathf.CeilToInt(r);
        for (int y = -ri; y <= ri; y++)
        for (int x = -ri; x <= ri; x++)
        {
            int px = Mathf.FloorToInt(p.x) + x, py = Mathf.FloorToInt(p.y) + y;
            if (px < 0 || py < 0 || px >= Size || py >= Size || x * x + y * y > r * r + 0.5f) continue;
            _pixels[py * Size + px] = c;
        }
    }

    void PushUndo()
    {
        _undo.Push((Color32[])_pixels.Clone());
        if (_undo.Count > 30) { var keep = _undo.ToArray(); _undo.Clear(); for (int i = 19; i >= 0; i--) _undo.Push(keep[i]); }
    }

    void Clear(bool undoable)
    {
        if (undoable) PushUndo();
        for (int i = 0; i < _pixels.Length; i++) _pixels[i] = Black;
        ApplyCanvas();
    }

    void ApplyCanvas()
    {
        _canvas.SetPixels32(_pixels);
        _canvas.Apply();
        Repaint();
    }

    void RandomSketch(int seed)
    {
        PushUndo();
        var tex = TerrainSketchModel.CreateRandomSketch(seed, _ridges, _rivers, _peaks);
        tex.GetPixels32().CopyTo(_pixels, 0);
        DestroyImmediate(tex);
        ApplyCanvas();
        if (_terrain != null && _modelAsset != null) Generate();
        _status = $"Seed {seed}: generated. Paint on top of it and press Generate to change it.\n" + _status;
    }

    void SaveCopy()
    {
        string folder = OutputFolder + "/Saved";
        if (!AssetDatabase.IsValidFolder(OutputFolder)) AssetDatabase.CreateFolder(Path.GetDirectoryName(OutputFolder).Replace('\\', '/'), Path.GetFileName(OutputFolder));
        if (!AssetDatabase.IsValidFolder(folder)) AssetDatabase.CreateFolder(OutputFolder, "Saved");
        string path = AssetDatabase.GenerateUniqueAssetPath($"{folder}/Sketch_{System.DateTime.Now:yyyyMMdd_HHmmss}.png");
        File.WriteAllBytes(path, _canvas.EncodeToPNG());
        AssetDatabase.ImportAsset(path);
        _status = $"Saved {path}. Use 'Load sketch...' to bring it back.";
        EditorGUIUtility.PingObject(AssetDatabase.LoadAssetAtPath<Texture2D>(path));
    }

    void LoadSketchFile()
    {
        string start = AssetDatabase.IsValidFolder(OutputFolder + "/Saved") ? OutputFolder + "/Saved" : OutputFolder;
        string path = EditorUtility.OpenFilePanel("Load sketch (256x256 PNG)", start, "png");
        if (string.IsNullOrEmpty(path)) return;
        var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
        if (!tex.LoadImage(File.ReadAllBytes(path)) || tex.width != Size || tex.height != Size)
        {
            _status = $"'{Path.GetFileName(path)}' is not a {Size}x{Size} PNG.";
            DestroyImmediate(tex);
            return;
        }
        PushUndo();
        tex.GetPixels32().CopyTo(_pixels, 0);
        DestroyImmediate(tex);
        ApplyCanvas();
        if (_terrain != null && _modelAsset != null) Generate();
    }

    void LoadFrom(NeuralTerrain t)
    {
        var src = t.sourceSketch;
        if (src.isReadable && src.width == Size && src.height == Size) { src.GetPixels32().CopyTo(_pixels, 0); ApplyCanvas(); }
        if (t.sourceHeightmap != null && t.sourceHeightmap.isReadable) ShowPreview(t.sourceHeightmap.GetPixelData<float>(0).ToArray());
    }

    // ------------------------------------------------------------------ generate

    void Generate()
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        _model ??= new TerrainSketchModel(_modelAsset);
        float[] heights = _model.Generate(_canvas);
        long modelMs = sw.ElapsedMilliseconds;

        ShowPreview(heights);
        SaveAndApply(heights);
        _status = $"Generated in {sw.ElapsedMilliseconds} ms (model {modelMs} ms). Saved to {OutputFolder}.";
        Repaint();
    }

    void ShowPreview(float[] heights)
    {
        var px = new Color32[heights.Length];
        for (int i = 0; i < px.Length; i++) { byte v = (byte)(Mathf.Clamp01(heights[i]) * 255f); px[i] = new Color32(v, v, v, 255); }
        _preview.SetPixels32(px);
        _preview.Apply();
    }

    void SaveAndApply(float[] heights)
    {
        if (!AssetDatabase.IsValidFolder(OutputFolder))
            AssetDatabase.CreateFolder(Path.GetDirectoryName(OutputFolder).Replace('\\', '/'), Path.GetFileName(OutputFolder));

        // Sketch -> PNG (readable, lossless, no sRGB so the bytes stay exactly what was painted).
        string sketchPath = OutputFolder + "/Sketch.png";
        File.WriteAllBytes(sketchPath, _canvas.EncodeToPNG());
        AssetDatabase.ImportAsset(sketchPath, ImportAssetOptions.ForceSynchronousImport);
        if (AssetImporter.GetAtPath(sketchPath) is TextureImporter ti && (!ti.isReadable || ti.sRGBTexture || ti.mipmapEnabled))
        {
            ti.isReadable = true; ti.sRGBTexture = false; ti.mipmapEnabled = false;
            ti.textureCompression = TextureImporterCompression.Uncompressed;
            ti.filterMode = FilterMode.Point; ti.npotScale = TextureImporterNPOTScale.None;
            ti.SaveAndReimport();
        }
        var sketchAsset = AssetDatabase.LoadAssetAtPath<Texture2D>(sketchPath);

        // Heightmap -> float texture asset (exact model values, 0..1).
        string heightPath = OutputFolder + "/Heightmap.asset";
        var heightAsset = AssetDatabase.LoadAssetAtPath<Texture2D>(heightPath);
        if (heightAsset == null)
        {
            heightAsset = new Texture2D(Size, Size, TextureFormat.RFloat, false, true) { name = "Heightmap", wrapMode = TextureWrapMode.Clamp };
            AssetDatabase.CreateAsset(heightAsset, heightPath);
        }
        heightAsset.SetPixelData(heights, 0);
        heightAsset.Apply();
        EditorUtility.SetDirty(heightAsset);
        AssetDatabase.SaveAssetIfDirty(heightAsset);

        if (_terrain == null) return;
        Undo.RecordObject(_terrain, "Generate Neural Terrain");
        _terrain.sourceSketch = sketchAsset;
        _terrain.sourceHeightmap = heightAsset;
        EditorUtility.SetDirty(_terrain);
        _terrain.Rebuild();
    }

    void RunVerify()
    {
        EditorUtility.DisplayProgressBar("Verify", "Running the model on GPU and on the CPU reference (takes ~20 s)...", 0.5f);
        try
        {
            DisposeModel();
            string report = TerrainSketchModel.Verify(_modelAsset, _canvas, out bool ok);
            if (ok) Debug.Log(report); else Debug.LogError(report);
            _status = report;
        }
        finally { EditorUtility.ClearProgressBar(); }
    }
}
