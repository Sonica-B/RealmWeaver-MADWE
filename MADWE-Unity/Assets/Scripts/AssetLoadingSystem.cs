using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine.Networking;

public class AssetLoadingSystem : MonoBehaviour
{
    [Header("Asset Paths")]
    public string textureBasePath = "StreamingAssets/Textures/";
    public string materialBasePath = "StreamingAssets/Materials/";
    
    [Header("Cache Settings")]
    public int maxCachedTextures = 100;
    public int maxCachedMaterials = 50;
    public bool preloadCommonAssets = true;
    
    [Header("Loading Settings")]
    public bool asyncLoading = true;
    public int maxConcurrentLoads = 3;
    
    private Dictionary<string, Texture2D> textureCache = new Dictionary<string, Texture2D>();
    private Dictionary<string, Material> materialCache = new Dictionary<string, Material>();
    private Queue<AssetLoadRequest> loadQueue = new Queue<AssetLoadRequest>();
    private int currentLoadCount = 0;
    
    // LRU cache tracking
    private LinkedList<string> textureLRU = new LinkedList<string>();
    private LinkedList<string> materialLRU = new LinkedList<string>();
    
    [System.Serializable]
    public class AssetLoadRequest
    {
        public string assetPath;
        public AssetType assetType;
        public System.Action<Object> callback;
    }
    
    public enum AssetType
    {
        Texture,
        Material,
        Mesh
    }
    
    void Start()
    {
        if (preloadCommonAssets)
        {
            StartCoroutine(PreloadAssets());
        }
        
        StartCoroutine(ProcessLoadQueue());
    }
    
    IEnumerator PreloadAssets()
    {
        // Preload common textures for each biome
        string[] biomes = { "forest", "desert", "snow" };
        string[] commonTextures = { "grass", "dirt", "stone", "water" };
        
        foreach (string biome in biomes)
        {
            foreach (string texture in commonTextures)
            {
                string path = $"{biome}/{texture}.png";
                yield return LoadTextureAsync(path, null);
            }
        }
        
        Debug.Log($"Preloaded {textureCache.Count} textures");
    }
    
    public void LoadTexture(string relativePath, System.Action<Texture2D> callback)
    {
        // Check cache first
        if (textureCache.ContainsKey(relativePath))
        {
            UpdateLRU(textureLRU, relativePath);
            callback?.Invoke(textureCache[relativePath]);
            return;
        }
        
        if (asyncLoading)
        {
            var request = new AssetLoadRequest
            {
                assetPath = relativePath,
                assetType = AssetType.Texture,
                callback = (obj) => callback?.Invoke(obj as Texture2D)
            };
            loadQueue.Enqueue(request);
        }
        else
        {
            StartCoroutine(LoadTextureAsync(relativePath, callback));
        }
    }
    
    IEnumerator LoadTextureAsync(string relativePath, System.Action<Texture2D> callback)
    {
        string fullPath = Path.Combine(Application.streamingAssetsPath, textureBasePath, relativePath);
        
        // Handle different platforms
        string loadPath = fullPath;
        if (Application.platform == RuntimePlatform.Android)
        {
            loadPath = "file://" + fullPath;
        }
        else if (Application.platform == RuntimePlatform.WindowsPlayer || 
                 Application.platform == RuntimePlatform.WindowsEditor)
        {
            loadPath = "file:///" + fullPath.Replace("\\", "/");
        }
        
        using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(loadPath))
        {
            yield return www.SendWebRequest();
            
            if (www.result == UnityWebRequest.Result.Success)
            {
                Texture2D texture = DownloadHandlerTexture.GetContent(www);
                texture.name = Path.GetFileNameWithoutExtension(relativePath);
                
                // Apply texture settings
                texture.wrapMode = TextureWrapMode.Repeat;
                texture.filterMode = FilterMode.Bilinear;
                texture.anisoLevel = 4;
                
                CacheTexture(relativePath, texture);
                callback?.Invoke(texture);
            }
            else
            {
                Debug.LogError($"Failed to load texture: {relativePath} - {www.error}");
                callback?.Invoke(null);
            }
        }
    }
    
    void CacheTexture(string path, Texture2D texture)
    {
        // Check cache size
        if (textureCache.Count >= maxCachedTextures)
        {
            // Remove least recently used
            string lru = textureLRU.First.Value;
            textureLRU.RemoveFirst();
            
            if (textureCache.ContainsKey(lru))
            {
                Destroy(textureCache[lru]);
                textureCache.Remove(lru);
            }
        }
        
        textureCache[path] = texture;
        textureLRU.AddLast(path);
    }
    
    public Material CreateMaterialFromTextures(string name, Dictionary<string, string> textureMap)
    {
        // Check cache
        if (materialCache.ContainsKey(name))
        {
            UpdateLRU(materialLRU, name);
            return materialCache[name];
        }
        
        // Create new material
        Material material = new Material(Shader.Find("Standard"));
        material.name = name;
        
        // Load and apply textures
        StartCoroutine(LoadMaterialTextures(material, textureMap, () =>
        {
            CacheMaterial(name, material);
        }));
        
        return material;
    }
    
    IEnumerator LoadMaterialTextures(Material material, Dictionary<string, string> textureMap, System.Action onComplete)
    {
        foreach (var kvp in textureMap)
        {
            string propertyName = kvp.Key;
            string texturePath = kvp.Value;
            
            bool textureLoaded = false;
            LoadTexture(texturePath, (texture) =>
            {
                if (texture != null && material != null)
                {
                    switch (propertyName)
                    {
                        case "_MainTex":
                        case "_BaseMap": // URP
                            material.mainTexture = texture;
                            break;
                        case "_BumpMap":
                        case "_NormalMap":
                            material.SetTexture("_BumpMap", texture);
                            material.EnableKeyword("_NORMALMAP");
                            break;
                        case "_MetallicGlossMap":
                            material.SetTexture("_MetallicGlossMap", texture);
                            material.EnableKeyword("_METALLICGLOSSMAP");
                            break;
                        case "_OcclusionMap":
                            material.SetTexture("_OcclusionMap", texture);
                            break;
                        default:
                            material.SetTexture(propertyName, texture);
                            break;
                    }
                }
                textureLoaded = true;
            });
            
            // Wait for texture to load
            while (!textureLoaded)
            {
                yield return null;
            }
        }
        
        onComplete?.Invoke();
    }
    
    void CacheMaterial(string name, Material material)
    {
        if (materialCache.Count >= maxCachedMaterials)
        {
            string lru = materialLRU.First.Value;
            materialLRU.RemoveFirst();
            
            if (materialCache.ContainsKey(lru))
            {
                Destroy(materialCache[lru]);
                materialCache.Remove(lru);
            }
        }
        
        materialCache[name] = material;
        materialLRU.AddLast(name);
    }
    
    IEnumerator ProcessLoadQueue()
    {
        while (true)
        {
            if (loadQueue.Count > 0 && currentLoadCount < maxConcurrentLoads)
            {
                var request = loadQueue.Dequeue();
                currentLoadCount++;
                
                switch (request.assetType)
                {
                    case AssetType.Texture:
                        StartCoroutine(LoadTextureWithCallback(request));
                        break;
                    case AssetType.Material:
                        // Handle material loading
                        break;
                }
            }
            
            yield return null;
        }
    }
    
    IEnumerator LoadTextureWithCallback(AssetLoadRequest request)
    {
        yield return LoadTextureAsync(request.assetPath, (texture) =>
        {
            request.callback?.Invoke(texture);
            currentLoadCount--;
        });
    }
    
    void UpdateLRU(LinkedList<string> lru, string key)
    {
        lru.Remove(key);
        lru.AddLast(key);
    }
    
    public void ClearCache()
    {
        foreach (var texture in textureCache.Values)
        {
            Destroy(texture);
        }
        textureCache.Clear();
        textureLRU.Clear();
        
        foreach (var material in materialCache.Values)
        {
            Destroy(material);
        }
        materialCache.Clear();
        materialLRU.Clear();
    }
    
    void OnDestroy()
    {
        ClearCache();
    }
    
    // API for Python communication
    public void LoadAssetFromData(string assetType, string assetName, byte[] data)
    {
        StartCoroutine(ProcessAssetData(assetType, assetName, data));
    }
    
    IEnumerator ProcessAssetData(string assetType, string assetName, byte[] data)
    {
        switch (assetType.ToLower())
        {
            case "texture":
                Texture2D texture = new Texture2D(2, 2);
                if (texture.LoadImage(data))
                {
                    CacheTexture(assetName, texture);
                    Debug.Log($"Loaded texture from data: {assetName}");
                }
                break;
                
            case "material":
                // Process material JSON data
                string json = System.Text.Encoding.UTF8.GetString(data);
                ProcessMaterialJSON(assetName, json);
                break;
        }
        
        yield return null;
    }
    
    void ProcessMaterialJSON(string name, string json)
    {
        // Parse material properties from JSON
        // This would be expanded based on your material format
        try
        {
            var materialData = JsonUtility.FromJson<MaterialData>(json);
            
            var textureMap = new Dictionary<string, string>();
            if (!string.IsNullOrEmpty(materialData.albedoTexture))
                textureMap["_MainTex"] = materialData.albedoTexture;
            if (!string.IsNullOrEmpty(materialData.normalTexture))
                textureMap["_BumpMap"] = materialData.normalTexture;
                
            CreateMaterialFromTextures(name, textureMap);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to parse material JSON: {e.Message}");
        }
    }
    
    [System.Serializable]
    public class MaterialData
    {
        public string albedoTexture;
        public string normalTexture;
        public string metallicTexture;
        public Color albedoColor = Color.white;
        public float metallic = 0f;
        public float smoothness = 0.5f;
    }
}