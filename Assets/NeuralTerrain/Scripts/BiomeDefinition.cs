using UnityEngine;

/// <summary>
/// Defines a single biome: the height/rainfall window it occupies on the
/// biome map, plus its visual identity (grass color). Extend this later with
/// spawn tables (monsters, plants, loot) as the devlog mentions.
///
/// Create instances via: Assets > Create > Terrain > Biome Definition
///
/// IMPORTANT: a biome's numeric index (used by the shader to pick its color)
/// is NOT stored here -- it's simply this asset's position in the
/// BiomeMap.biomes list. See BiomeMap.cs for why, and keep that in mind if
/// you ever reorder biomes in the Inspector after terrain has been baked.
/// </summary>
[CreateAssetMenu(fileName = "NewBiome", menuName = "Terrain/Biome Definition")]
public class BiomeDefinition : ScriptableObject
{
    [Header("Identity")]
    public string biomeName = "New Biome";

    [Header("Placement rules")]
    [Tooltip("Normalized height range [0-1], as a fraction of BiomeMap.maxWorldHeight, where this biome can appear.")]
    public Vector2 heightRange = new Vector2(0f, 1f);

    [Tooltip("Rainfall range [0-1] where this biome can appear (from Perlin noise, so effectively 'moisture').")]
    public Vector2 rainfallRange = new Vector2(0f, 1f);

    [Header("Visuals")]
    public Color grassColor = Color.green;

    /// <summary>
    /// True if a point with this normalized height and rainfall falls inside
    /// this biome's defined window.
    /// </summary>
    public bool Matches(float normalizedHeight, float rainfall)
    {
        return normalizedHeight >= heightRange.x && normalizedHeight <= heightRange.y &&
               rainfall >= rainfallRange.x && rainfall <= rainfallRange.y;
    }
}
