using UnityEngine;
using System.Collections.Generic;

public class TileManager : MonoBehaviour
{
    public TilePrefabManager prefabManager;
    private Dictionary<string, GameObject> chunkContainers = new Dictionary<string, GameObject>();
    
    void Start()
    {
        // Find prefab manager if not assigned
        if (prefabManager == null)
        {
            prefabManager = GetComponent<TilePrefabManager>();
            if (prefabManager == null)
            {
                prefabManager = gameObject.AddComponent<TilePrefabManager>();
            }
        }
    }
    
    public void CreateTileGrid(int[,] tileData, string biome)
    {
        // Backward compatibility - calls the new method with zero offset
        CreateTileGridAtPosition(tileData, biome, Vector3.zero);
    }
    
    public void CreateTileGridAtPosition(int[,] tileData, string biome, Vector3 offset)
    {
        // Create container for this chunk
        string chunkName = $"Chunk_{biome}_{offset.x}_{offset.z}";
        
        // Clean up old chunk if exists
        if (chunkContainers.ContainsKey(chunkName))
        {
            Destroy(chunkContainers[chunkName]);
        }
        
        GameObject chunkContainer = new GameObject(chunkName);
        chunkContainer.transform.position = offset;
        chunkContainers[chunkName] = chunkContainer;
        
        int height = tileData.GetLength(0);
        int width = tileData.GetLength(1);
        
        // Create tiles using prefabs
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Vector3 tilePosition = offset + new Vector3(x, 0, y);
                
                // Get prefab from manager
                GameObject tile = prefabManager.GetTilePrefab(biome, tileData[y, x], tilePosition);
                
                if (tile != null)
                {
                    tile.transform.parent = chunkContainer.transform;
                    tile.name = $"Tile_{x}_{y}";
                }
            }
        }
        
        Debug.Log($"Created {width}x{height} {biome} grid with prefabs at {offset}");
        
        // Ensure camera can see tiles
        EnsureCameraView(offset, width, height);
    }
    
    void EnsureCameraView(Vector3 chunkCenter, int width, int height)
    {
        Camera mainCamera = Camera.main;
        if (mainCamera == null)
        {
            GameObject cameraObj = new GameObject("Main Camera");
            mainCamera = cameraObj.AddComponent<Camera>();
            mainCamera.tag = "MainCamera";
            cameraObj.AddComponent<AudioListener>();
        }
        
        // Position camera to see all chunks
        float viewDistance = Mathf.Max(width, height) * 1.5f;
        mainCamera.transform.position = new Vector3(
            chunkCenter.x + width / 2f,
            viewDistance,
            chunkCenter.z - viewDistance * 0.5f
        );
        mainCamera.transform.LookAt(chunkCenter + new Vector3(width / 2f, 0, height / 2f));
    }
}