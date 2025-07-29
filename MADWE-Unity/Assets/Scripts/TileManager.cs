using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using System;

public class TileManager : MonoBehaviour
{
    [Header("Prefab Configuration")]
    public GameObject[] forestPrefabs;
    public GameObject[] desertPrefabs;
    public GameObject[] cyberpunkPrefabs;
    
    [Header("Connection Settings")]
    public string pythonHost = "127.0.0.1";
    public int pythonPort = 5005;
    
    [Header("Generation Settings")]
    public float tileSize = 1.0f;
    public Transform worldContainer;
    
    private TcpClient client;
    private NetworkStream stream;
    private Thread receiveThread;
    private bool isConnected = false;
    
    // Tile cache
    private Dictionary<string, GameObject> prefabCache = new Dictionary<string, GameObject>();
    private Dictionary<Vector2Int, GameObject> spawnedTiles = new Dictionary<Vector2Int, GameObject>();
    
    // Message queue
    private Queue<string> messageQueue = new Queue<string>();
    private object queueLock = new object();
    
    void Start()
    {
        InitializePrefabCache();
        ConnectToPython();
    }
    
    void InitializePrefabCache()
    {
        // Forest biome
        prefabCache["Tiles/Forest/Grass"] = forestPrefabs[0];
        prefabCache["Tiles/Forest/Tree"] = forestPrefabs[1];
        prefabCache["Tiles/Forest/Path"] = forestPrefabs[2];
        prefabCache["Tiles/Forest/Water"] = forestPrefabs[3];
        
        // Desert biome
        prefabCache["Tiles/Desert/Sand"] = desertPrefabs[0];
        prefabCache["Tiles/Desert/Dune"] = desertPrefabs[1];
        prefabCache["Tiles/Desert/Cactus"] = desertPrefabs[2];
        prefabCache["Tiles/Desert/Oasis"] = desertPrefabs[3];
        
        // Cyberpunk biome
        prefabCache["Tiles/Cyberpunk/Street"] = cyberpunkPrefabs[0];
        prefabCache["Tiles/Cyberpunk/Building"] = cyberpunkPrefabs[1];
        prefabCache["Tiles/Cyberpunk/Neon"] = cyberpunkPrefabs[2];
        prefabCache["Tiles/Cyberpunk/Plaza"] = cyberpunkPrefabs[3];
    }
    
    void ConnectToPython()
    {
        try
        {
            client = new TcpClient(pythonHost, pythonPort);
            stream = client.GetStream();
            isConnected = true;
            
            // Start receive thread
            receiveThread = new Thread(ReceiveMessages);
            receiveThread.Start();
            
            // Send initial connection message
            SendMessage(new ConnectionMessage { type = "unity_connected" });
            
            Debug.Log($"Connected to Python bridge at {pythonHost}:{pythonPort}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to connect to Python: {e.Message}");
        }
    }
    
    void ReceiveMessages()
    {
        byte[] buffer = new byte[4096];
        
        while (isConnected && stream.CanRead)
        {
            try
            {
                // Read length prefix
                byte[] lengthBuffer = new byte[4];
                int bytesRead = stream.Read(lengthBuffer, 0, 4);
                if (bytesRead == 0) break;
                
                uint messageLength = BitConverter.ToUInt32(lengthBuffer, 0);
                if (BitConverter.IsLittleEndian)
                    messageLength = ReverseBytes(messageLength);
                
                // Read message
                byte[] messageBuffer = new byte[messageLength];
                int totalRead = 0;
                while (totalRead < messageLength)
                {
                    int read = stream.Read(messageBuffer, totalRead, (int)(messageLength - totalRead));
                    if (read == 0) break;
                    totalRead += read;
                }
                
                string message = Encoding.UTF8.GetString(messageBuffer);
                
                // Queue message for main thread
                lock (queueLock)
                {
                    messageQueue.Enqueue(message);
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Receive error: {e.Message}");
                break;
            }
        }
    }
    
    void Update()
    {
        // Process queued messages on main thread
        lock (queueLock)
        {
            while (messageQueue.Count > 0)
            {
                string message = messageQueue.Dequeue();
                ProcessMessage(message);
            }
        }
        
        // Request generation on key press (for testing)
        if (Input.GetKeyDown(KeyCode.G))
        {
            RequestChunkGeneration(Vector2Int.zero);
        }
    }
    
    void ProcessMessage(string jsonMessage)
    {
        try
        {
            var message = JsonConvert.DeserializeObject<Dictionary<string, object>>(jsonMessage);
            string msgType = message["type"].ToString();
            
            switch (msgType)
            {
                case "chunk_generated":
                    ProcessChunkData(message["data"] as Newtonsoft.Json.Linq.JObject);
                    break;
                case "tile_update":
                    ProcessTileUpdate(message["data"] as Newtonsoft.Json.Linq.JObject);
                    break;
                case "performance_stats":
                    LogPerformanceStats(message["data"] as Newtonsoft.Json.Linq.JObject);
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to process message: {e.Message}");
        }
    }
    
    void ProcessChunkData(Newtonsoft.Json.Linq.JObject data)
    {
        // Extract chunk information
        int width = data["width"].Value<int>();
        int height = data["height"].Value<int>();
        var tiles = data["tiles"].ToObject<List<TileData>>();
        
        // Clear existing tiles (optional)
        ClearChunk();
        
        // Spawn tiles
        foreach (var tile in tiles)
        {
            SpawnTile(tile);
        }
        
        Debug.Log($"Generated {tiles.Count} tiles in {width}x{height} chunk");
    }
    
    void SpawnTile(TileData tile)
    {
        if (!prefabCache.ContainsKey(tile.prefab))
        {
            Debug.LogWarning($"Prefab not found: {tile.prefab}");
            return;
        }
        
        Vector3 position = new Vector3(tile.x * tileSize, 0, tile.y * tileSize);
        Quaternion rotation = Quaternion.Euler(0, tile.rotation, 0);
        
        GameObject tilePrefab = prefabCache[tile.prefab];
        GameObject instance = Instantiate(tilePrefab, position, rotation, worldContainer);
        
        // Store reference
        Vector2Int gridPos = new Vector2Int(tile.x, tile.y);
        if (spawnedTiles.ContainsKey(gridPos))
        {
            Destroy(spawnedTiles[gridPos]);
        }
        spawnedTiles[gridPos] = instance;
        
        // Add tile metadata
        var tileComponent = instance.AddComponent<TileInfo>();
        tileComponent.Initialize(tile);
    }
    
    void ClearChunk()
    {
        foreach (var kvp in spawnedTiles)
        {
            if (kvp.Value != null)
                Destroy(kvp.Value);
        }
        spawnedTiles.Clear();
    }
    
    public void RequestChunkGeneration(Vector2Int chunkPosition)
    {
        var request = new ChunkRequest
        {
            type = "generate_chunk",
            data = new ChunkRequestData
            {
                position = new int[] { chunkPosition.x, chunkPosition.y },
                biome = "forest",
                size = new int[] { 32, 32 }
            }
        };
        
        SendMessage(request);
    }
    
    void SendMessage(object message)
    {
        if (!isConnected || !stream.CanWrite) return;
        
        try
        {
            string json = JsonConvert.SerializeObject(message);
            byte[] data = Encoding.UTF8.GetBytes(json);
            
            // Send length prefix
            byte[] lengthPrefix = BitConverter.GetBytes((uint)data.Length);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(lengthPrefix);
            
            stream.Write(lengthPrefix, 0, 4);
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to send message: {e.Message}");
        }
    }
    
    void LogPerformanceStats(Newtonsoft.Json.Linq.JObject stats)
    {
        float latency = stats["avg_latency"].Value<float>();
        float fps = stats["generation_fps"].Value<float>();
        Debug.Log($"Performance - Latency: {latency:F1}ms, Generation FPS: {fps:F1}");
    }
    
    uint ReverseBytes(uint value)
    {
        return (value & 0x000000FFU) << 24 | (value & 0x0000FF00U) << 8 |
               (value & 0x00FF0000U) >> 8 | (value & 0xFF000000U) >> 24;
    }
    
    void OnDestroy()
    {
        isConnected = false;
        
        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(1000);
        }
        
        if (stream != null) stream.Close();
        if (client != null) client.Close();
    }
    
    // Data structures
    [System.Serializable]
    public class TileData
    {
        public int x;
        public int y;
        public int id;
        public string name;
        public string biome;
        public float rotation;
        public string prefab;
    }
    
    [System.Serializable]
    public class ConnectionMessage
    {
        public string type;
        public long timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds();
    }
    
    [System.Serializable]
    public class ChunkRequest
    {
        public string type;
        public ChunkRequestData data;
    }
    
    [System.Serializable]
    public class ChunkRequestData
    {
        public int[] position;
        public string biome;
        public int[] size;
    }
}

// Tile information component
public class TileInfo : MonoBehaviour
{
    public int tileId;
    public string tileName;
    public string biome;
    public Vector2Int gridPosition;
    
    public void Initialize(TileManager.TileData data)
    {
        tileId = data.id;
        tileName = data.name;
        biome = data.biome;
        gridPosition = new Vector2Int(data.x, data.y);
    }
}