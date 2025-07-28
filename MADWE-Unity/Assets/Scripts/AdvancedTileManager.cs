using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public class AdvancedTileManager : MonoBehaviour
{
    [Header("Managers")]
    public TilePrefabManager prefabManager;
    
    [Header("Tile Settings")]
    public float tileSize = 1.0f;
    public float tileGap = 0.0f;
    public bool smoothChunkBoundaries = true;
    
    [Header("LOD Settings")]
    public bool enableLOD = true;
    public float[] lodDistances = { 20f, 50f, 100f };
    public float lodUpdateInterval = 0.5f;
    
    [Header("Performance")]
    public int maxTilesPerFrame = 50;
    public bool asyncPlacement = true;
    
    private Dictionary<Vector2Int, ChunkData> loadedChunks = new Dictionary<Vector2Int, ChunkData>();
    private Transform chunksContainer;
    private Camera mainCamera;
    private Coroutine lodUpdateCoroutine;
    
    [System.Serializable]
    public class ChunkData
    {
        public Vector2Int chunkCoord;
        public string biome;
        public List<GameObject> tiles = new List<GameObject>();
        public Bounds bounds;
        public float lastLODUpdate;
    }
    
    void Awake()
    {
        chunksContainer = new GameObject("Chunks").transform;
        chunksContainer.parent = transform;
        
        if (prefabManager == null)
        {
            prefabManager = GetComponent<TilePrefabManager>();
            if (prefabManager == null)
            {
                prefabManager = gameObject.AddComponent<TilePrefabManager>();
            }
        }
        
        mainCamera = Camera.main;
        if (mainCamera == null)
        {
            Debug.LogWarning("No main camera found for LOD calculations");
        }
    }
    
    void Start()
    {
        if (enableLOD && mainCamera != null)
        {
            lodUpdateCoroutine = StartCoroutine(UpdateLODSystem());
        }
    }
    
    public void CreateTileGridAtPosition(int[,] tileData, string biome, Vector3 offset)
    {
        Vector2Int chunkCoord = new Vector2Int(
            Mathf.RoundToInt(offset.x / tileSize),
            Mathf.RoundToInt(offset.z / tileSize)
        );
        
        // Clear existing chunk
        if (loadedChunks.ContainsKey(chunkCoord))
        {
            ClearChunk(chunkCoord);
        }
        
        // Start placement
        if (asyncPlacement)
        {
            StartCoroutine(PlaceTilesAsync(tileData, biome, offset, chunkCoord));
        }
        else
        {
            PlaceTilesImmediate(tileData, biome, offset, chunkCoord);
        }
    }
    
    void PlaceTilesImmediate(int[,] tileData, string biome, Vector3 offset, Vector2Int chunkCoord)
    {
        ChunkData chunk = CreateChunk(chunkCoord, biome, offset, tileData.GetLength(1), tileData.GetLength(0));
        
        int height = tileData.GetLength(0);
        int width = tileData.GetLength(1);
        
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Vector3 localPos = GetTilePosition(x, y);
                Vector3 worldPos = offset + localPos;
                
                GameObject tile = PlaceTile(tileData[y, x], biome, worldPos, chunk);
                
                // Apply chunk boundary smoothing
                if (smoothChunkBoundaries && IsChunkBoundaryTile(x, y, width, height))
                {
                    ApplyBoundarySmoothing(tile, x, y, width, height);
                }
            }
        }
        
        loadedChunks[chunkCoord] = chunk;
    }
    
    IEnumerator PlaceTilesAsync(int[,] tileData, string biome, Vector3 offset, Vector2Int chunkCoord)
    {
        ChunkData chunk = CreateChunk(chunkCoord, biome, offset, tileData.GetLength(1), tileData.GetLength(0));
        
        int height = tileData.GetLength(0);
        int width = tileData.GetLength(1);
        int tilesPlaced = 0;
        
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Vector3 localPos = GetTilePosition(x, y);
                Vector3 worldPos = offset + localPos;
                
                GameObject tile = PlaceTile(tileData[y, x], biome, worldPos, chunk);
                
                if (smoothChunkBoundaries && IsChunkBoundaryTile(x, y, width, height))
                {
                    ApplyBoundarySmoothing(tile, x, y, width, height);
                }
                
                tilesPlaced++;
                
                // Yield periodically to avoid frame drops
                if (tilesPlaced % maxTilesPerFrame == 0)
                {
                    yield return null;
                }
            }
        }
        
        loadedChunks[chunkCoord] = chunk;
        Debug.Log($"Async placement complete for chunk {chunkCoord} ({tilesPlaced} tiles)");
    }
    
    ChunkData CreateChunk(Vector2Int coord, string biome, Vector3 offset, int width, int height)
    {
        GameObject chunkObj = new GameObject($"Chunk_{coord.x}_{coord.y}_{biome}");
        chunkObj.transform.parent = chunksContainer;
        chunkObj.transform.position = offset;
        
        ChunkData chunk = new ChunkData
        {
            chunkCoord = coord,
            biome = biome,
            bounds = new Bounds(
                offset + new Vector3(width * tileSize * 0.5f, 0, height * tileSize * 0.5f),
                new Vector3(width * tileSize, 5f, height * tileSize)
            )
        };
        
        return chunk;
    }
    
    GameObject PlaceTile(int tileId, string biome, Vector3 position, ChunkData chunk)
    {
        GameObject tile = prefabManager.GetTilePrefab(biome, tileId, position);
        
        if (tile != null)
        {
            tile.transform.parent = chunksContainer.Find($"Chunk_{chunk.chunkCoord.x}_{chunk.chunkCoord.y}_{biome}");
            chunk.tiles.Add(tile);
            
            // Add LOD component if enabled
            if (enableLOD && tile.GetComponent<TileLOD>() == null)
            {
                tile.AddComponent<TileLOD>();
            }
        }
        
        return tile;
    }
    
    Vector3 GetTilePosition(int x, int y)
    {
        float actualTileSize = tileSize + tileGap;
        return new Vector3(x * actualTileSize, 0, y * actualTileSize);
    }
    
    bool IsChunkBoundaryTile(int x, int y, int width, int height)
    {
        return x == 0 || x == width - 1 || y == 0 || y == height - 1;
    }
    
    void ApplyBoundarySmoothing(GameObject tile, int x, int y, int width, int height)
    {
        // Add slight height variation at boundaries to blend chunks
        float smoothingFactor = 0.1f;
        float heightOffset = 0;
        
        if (x == 0 || x == width - 1)
        {
            heightOffset += Mathf.Sin(y * 0.5f) * smoothingFactor;
        }
        if (y == 0 || y == height - 1)
        {
            heightOffset += Mathf.Cos(x * 0.5f) * smoothingFactor;
        }
        
        tile.transform.position += Vector3.up * heightOffset;
    }
    
    void ClearChunk(Vector2Int chunkCoord)
    {
        if (loadedChunks.ContainsKey(chunkCoord))
        {
            ChunkData chunk = loadedChunks[chunkCoord];
            
            foreach (var tile in chunk.tiles)
            {
                if (prefabManager.useObjectPooling)
                {
                    prefabManager.ReturnToPool(tile);
                }
                else
                {
                    Destroy(tile);
                }
            }
            
            GameObject chunkObj = chunksContainer.Find($"Chunk_{chunk.chunkCoord.x}_{chunk.chunkCoord.y}_{chunk.biome}")?.gameObject;
            if (chunkObj != null)
            {
                Destroy(chunkObj);
            }
            
            loadedChunks.Remove(chunkCoord);
        }
    }
    
    IEnumerator UpdateLODSystem()
    {
        while (enableLOD)
        {
            if (mainCamera != null)
            {
                Vector3 cameraPos = mainCamera.transform.position;
                
                foreach (var chunk in loadedChunks.Values)
                {
                    float distance = Vector3.Distance(cameraPos, chunk.bounds.center);
                    UpdateChunkLOD(chunk, distance);
                }
            }
            
            yield return new WaitForSeconds(lodUpdateInterval);
        }
    }
    
    void UpdateChunkLOD(ChunkData chunk, float distanceToCamera)
    {
        int targetLOD = 0;
        
        for (int i = 0; i < lodDistances.Length; i++)
        {
            if (distanceToCamera > lodDistances[i])
            {
                targetLOD = i + 1;
            }
        }
        
        foreach (var tile in chunk.tiles)
        {
            if (tile != null && tile.activeInHierarchy)
            {
                var lodComponent = tile.GetComponent<TileLOD>();
                if (lodComponent != null)
                {
                    lodComponent.SetLOD(targetLOD);
                }
            }
        }
    }
    
    public ChunkData GetChunk(Vector2Int coord)
    {
        return loadedChunks.ContainsKey(coord) ? loadedChunks[coord] : null;
    }
    
    public List<ChunkData> GetLoadedChunks()
    {
        return new List<ChunkData>(loadedChunks.Values);
    }
}