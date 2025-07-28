using UnityEngine;
using System.Collections.Generic;
using System.Linq;

[System.Serializable]
public class TilePrefabSet
{
    public string biome;
    public TilePrefabMapping[] mappings;
}

[System.Serializable]
public class TilePrefabMapping
{
    public int tileId;
    public string tileName;
    public GameObject[] prefabVariations;
    [Range(0f, 1f)]
    public float probability = 1f;
}

public class TilePrefabManager : MonoBehaviour
{
    [Header("Prefab Configuration")]
    public TilePrefabSet[] biomePrefabSets;
    
    [Header("Fallback Settings")]
    public GameObject defaultTilePrefab;
    public Material[] biomeMaterials;
    
    [Header("Performance")]
    public int poolSizePerPrefab = 50;
    public bool useObjectPooling = true;
    
    private Dictionary<string, Dictionary<int, TilePrefabMapping>> prefabLookup;
    private Dictionary<GameObject, Queue<GameObject>> objectPools;
    private Transform poolContainer;
    
    void Awake()
    {
        InitializePrefabSystem();
        if (useObjectPooling)
        {
            InitializeObjectPools();
        }
    }
    
    void InitializePrefabSystem()
    {
        prefabLookup = new Dictionary<string, Dictionary<int, TilePrefabMapping>>();
        
        foreach (var biomeSet in biomePrefabSets)
        {
            var biomeDict = new Dictionary<int, TilePrefabMapping>();
            foreach (var mapping in biomeSet.mappings)
            {
                biomeDict[mapping.tileId] = mapping;
            }
            prefabLookup[biomeSet.biome.ToLower()] = biomeDict;
        }
        
        // Create default prefab if missing
        if (defaultTilePrefab == null)
        {
            defaultTilePrefab = CreateDefaultPrefab();
        }
    }
    
    GameObject CreateDefaultPrefab()
    {
        GameObject prefab = GameObject.CreatePrimitive(PrimitiveType.Cube);
        prefab.name = "DefaultTilePrefab";
        
        // Add components
        var collider = prefab.GetComponent<BoxCollider>();
        if (collider == null)
        {
            collider = prefab.AddComponent<BoxCollider>();
        }
        
        // Setup renderer
        var renderer = prefab.GetComponent<MeshRenderer>();
        if (renderer != null)
        {
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
            renderer.receiveShadows = true;
        }
        
        prefab.SetActive(false);
        return prefab;
    }
    
    void InitializeObjectPools()
    {
        objectPools = new Dictionary<GameObject, Queue<GameObject>>();
        poolContainer = new GameObject("ObjectPools").transform;
        poolContainer.parent = transform;
        
        // Create pools for all prefab variations
        foreach (var biomeSet in biomePrefabSets)
        {
            foreach (var mapping in biomeSet.mappings)
            {
                foreach (var prefab in mapping.prefabVariations)
                {
                    if (prefab != null && !objectPools.ContainsKey(prefab))
                    {
                        CreatePool(prefab, poolSizePerPrefab);
                    }
                }
            }
        }
        
        // Create pool for default prefab
        if (!objectPools.ContainsKey(defaultTilePrefab))
        {
            CreatePool(defaultTilePrefab, poolSizePerPrefab * 2);
        }
    }
    
    void CreatePool(GameObject prefab, int size)
    {
        var pool = new Queue<GameObject>();
        var container = new GameObject($"Pool_{prefab.name}").transform;
        container.parent = poolContainer;
        
        for (int i = 0; i < size; i++)
        {
            var instance = Instantiate(prefab, container);
            instance.SetActive(false);
            pool.Enqueue(instance);
        }
        
        objectPools[prefab] = pool;
    }
    
    public GameObject GetTilePrefab(string biome, int tileId, Vector3 position)
    {
        GameObject prefab = null;
        
        // Try to find specific prefab for this biome and tile
        if (prefabLookup.ContainsKey(biome.ToLower()))
        {
            var biomeDict = prefabLookup[biome.ToLower()];
            if (biomeDict.ContainsKey(tileId))
            {
                var mapping = biomeDict[tileId];
                if (mapping.prefabVariations.Length > 0)
                {
                    // Select variation based on position (deterministic randomness)
                    int variationIndex = GetVariationIndex(position, mapping.prefabVariations.Length);
                    prefab = mapping.prefabVariations[variationIndex];
                }
            }
        }
        
        // Fallback to default
        if (prefab == null)
        {
            prefab = defaultTilePrefab;
        }
        
        // Get from pool or instantiate
        GameObject instance = useObjectPooling ? GetFromPool(prefab) : Instantiate(prefab);
        instance.transform.position = position;
        instance.SetActive(true);
        
        // Apply biome-specific modifications
        ApplyBiomeStyle(instance, biome, tileId);
        
        return instance;
    }
    
    GameObject GetFromPool(GameObject prefab)
    {
        if (objectPools.ContainsKey(prefab) && objectPools[prefab].Count > 0)
        {
            return objectPools[prefab].Dequeue();
        }
        
        // Pool empty, create new instance
        return Instantiate(prefab);
    }
    
    public void ReturnToPool(GameObject instance)
    {
        if (!useObjectPooling) 
        {
            Destroy(instance);
            return;
        }
        
        instance.SetActive(false);
        
        // Find which prefab this instance belongs to
        foreach (var kvp in objectPools)
        {
            if (instance.name.StartsWith(kvp.Key.name))
            {
                kvp.Value.Enqueue(instance);
                instance.transform.parent = poolContainer;
                return;
            }
        }
        
        // Couldn't find pool, destroy
        Destroy(instance);
    }
    
    int GetVariationIndex(Vector3 position, int variationCount)
    {
        // Deterministic variation based on position
        int hash = (int)(position.x * 73 + position.z * 97) % 1000;
        return Mathf.Abs(hash) % variationCount;
    }
    
    void ApplyBiomeStyle(GameObject tile, string biome, int tileId)
    {
        var renderer = tile.GetComponentInChildren<MeshRenderer>();
        if (renderer == null) return;
        
        // Apply material if using defaults
        if (tile == defaultTilePrefab || renderer.sharedMaterial == null)
        {
            int materialIndex = biome.ToLower() switch
            {
                "forest" => 0,
                "desert" => 1,
                "snow" => 2,
                _ => 0
            };
            
            if (biomeMaterials.Length > materialIndex)
            {
                renderer.material = biomeMaterials[materialIndex];
            }
        }
        
        // Apply color tinting based on tile type
        if (renderer.material != null)
        {
            Color tint = GetTileColor(biome, tileId);
            renderer.material.SetColor("_Color", tint);
            
            // Also try _BaseColor for URP/HDRP
            if (renderer.material.HasProperty("_BaseColor"))
            {
                renderer.material.SetColor("_BaseColor", tint);
            }
        }
    }
    
    Color GetTileColor(string biome, int tileType)
    {
        switch (biome.ToLower())
        {
            case "forest":
                return tileType switch
                {
                    0 => new Color(0.4f, 0.7f, 0.3f), // Grass
                    1 => new Color(0.3f, 0.2f, 0.1f), // Tree base
                    2 => new Color(0.6f, 0.5f, 0.4f), // Path
                    3 => new Color(0.3f, 0.5f, 0.7f), // Water
                    _ => Color.white
                };
            case "desert":
                return tileType switch
                {
                    10 => new Color(0.9f, 0.8f, 0.6f), // Sand
                    11 => new Color(0.8f, 0.7f, 0.5f), // Dune
                    12 => new Color(0.6f, 0.5f, 0.4f), // Rock
                    13 => new Color(0.2f, 0.6f, 0.8f), // Oasis
                    _ => Color.white
                };
            case "snow":
                return tileType switch
                {
                    20 => new Color(0.95f, 0.95f, 1.0f), // Snow
                    21 => new Color(0.7f, 0.85f, 0.95f), // Ice
                    22 => new Color(0.2f, 0.4f, 0.2f), // Pine
                    _ => Color.white
                };
            default:
                return Color.white;
        }
    }
}