using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System;
using System.IO;

public class MADWEController : MonoBehaviour
{
    private TcpClient client;
    private Thread receiveThread;
    private NetworkStream stream;
    private bool connected = false;
    private float lastConnectionAttempt = 0;
    
    [Header("Managers")]
    public AdvancedTileManager tileManager;
    public AssetLoadingSystem assetLoader;
    
    [System.Serializable]
    public class UnityMessage
    {
        public string type;
        public MessageData data;
        public float timestamp;
    }
    
    [System.Serializable]
    public class MessageData
    {
        public string status;
        public string message;
        public float sent_time;
    }
    
    [System.Serializable]
    public class TileUpdateData
    {
        public string type;
        public TileData data;
        public float timestamp;
    }
    
    [System.Serializable]
    public class TileData
    {
        public List<List<int>> tiles;
        public List<int> position;
        public List<int> size;
        public string biome;
    }
    
    [System.Serializable]
    public class AssetUpdateData
    {
        public string type;
        public AssetData data;
        public float timestamp;
    }
    
    [System.Serializable]
    public class AssetData
    {
        public string asset_type;
        public string asset_name;
        public string asset_data;
        public AssetMetadata metadata;
    }
    
    [System.Serializable]
    public class AssetMetadata
    {
        public string biome;
        public string name;
        public int size;
    }
    
    void Awake()
    {
        DontDestroyOnLoad(gameObject);
        UnityMainThreadDispatcher.Instance();
        
        // Find managers
        if (tileManager == null)
        {
            tileManager = FindFirstObjectByType<AdvancedTileManager>();
            if (tileManager == null)
            {
                GameObject managerObj = new GameObject("TileManager");
                tileManager = managerObj.AddComponent<AdvancedTileManager>();
            }
        }
        
        if (assetLoader == null)
        {
            assetLoader = FindFirstObjectByType<AssetLoadingSystem>();
            if (assetLoader == null)
            {
                assetLoader = gameObject.AddComponent<AssetLoadingSystem>();
            }
        }
    }
    
    void Start()
    {
        ConnectToPython();
    }
    
    void Update()
    {
        if (!connected && Time.time - lastConnectionAttempt > 5.0f)
        {
            lastConnectionAttempt = Time.time;
            ConnectToPython();
        }
    }
    
    void ConnectToPython()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 5005);
            stream = client.GetStream();
            connected = true;
            
            receiveThread = new Thread(ReceiveData);
            receiveThread.Start();
            
            Debug.Log("Connected to Python server!");
            SendMessageToPython("connection", "Unity client connected - Day 4 Integration");
        }
        catch (Exception e)
        {
            Debug.LogWarning($"Connection failed: {e.Message}. Retrying in 5 seconds...");
            connected = false;
        }
    }
    
    void SendMessageToPython(string messageType, string messageContent)
    {
        if (!connected || stream == null) return;
        
        try
        {
            var msg = new UnityMessage
            {
                type = messageType,
                data = new MessageData 
                { 
                    status = "ok", 
                    message = messageContent,
                    sent_time = Time.time
                },
                timestamp = Time.time
            };
            
            string json = JsonUtility.ToJson(msg) + "\n";
            byte[] data = Encoding.UTF8.GetBytes(json);
            stream.Write(data, 0, data.Length);
            stream.Flush();
        }
        catch (Exception e)
        {
            Debug.LogError($"Send error: {e.Message}");
            connected = false;
        }
    }
    
    void ReceiveData()
    {
        byte[] buffer = new byte[65536]; // Larger buffer for asset data
        string messageBuffer = "";
        
        while (connected)
        {
            try
            {
                if (stream.DataAvailable)
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead > 0)
                    {
                        messageBuffer += Encoding.UTF8.GetString(buffer, 0, bytesRead);
                        
                        string[] lines = messageBuffer.Split('\n');
                        for (int i = 0; i < lines.Length - 1; i++)
                        {
                            if (!string.IsNullOrEmpty(lines[i]))
                            {
                                string json = lines[i];
                                UnityMainThreadDispatcher.Instance().Enqueue(() => ProcessMessage(json));
                            }
                        }
                        messageBuffer = lines[lines.Length - 1];
                    }
                }
                Thread.Sleep(10);
            }
            catch (Exception e)
            {
                Debug.LogError($"Receive error: {e.Message}");
                connected = false;
            }
        }
    }
    
    void ProcessMessage(string json)
    {
        try
        {
            // First check message type
            var baseMsg = JsonUtility.FromJson<UnityMessage>(json);
            Debug.Log($"Received message type: {baseMsg.type}");
            
            switch (baseMsg.type)
            {
                case "tile_update":
                    ProcessTileUpdate(json);
                    break;
                    
                case "asset_update":
                    ProcessAssetUpdate(json);
                    break;
                    
                case "test":
                    SendMessageToPython("test_response", "Test received");
                    break;
                    
                default:
                    Debug.LogWarning($"Unknown message type: {baseMsg.type}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Process message error: {e.Message}\nJSON: {json}");
        }
    }
    
    void ProcessTileUpdate(string json)
    {
        try
        {
            TileUpdateData tileData = JsonUtility.FromJson<TileUpdateData>(json);
            
            if (tileManager != null && tileData.data != null)
            {
                int height = tileData.data.tiles.Count;
                int width = height > 0 ? tileData.data.tiles[0].Count : 0;
                int[,] tileArray = new int[height, width];
                
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        tileArray[y, x] = tileData.data.tiles[y][x];
                    }
                }
                
                Vector3 offset = new Vector3(
                    tileData.data.position[0], 
                    0, 
                    tileData.data.position[1]
                );
                
                tileManager.CreateTileGridAtPosition(tileArray, tileData.data.biome, offset);
                Debug.Log($"Created {width}x{height} {tileData.data.biome} grid at {offset}");
                
                // Send confirmation
                SendMessageToPython("tile_update_complete", 
                    $"Rendered {width}x{height} tiles at {offset}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Process tile update error: {e.Message}");
        }
    }
    
    void ProcessAssetUpdate(string json)
    {
        try
        {
            AssetUpdateData assetData = JsonUtility.FromJson<AssetUpdateData>(json);
            
            if (assetLoader != null && assetData.data != null)
            {
                Debug.Log($"Processing {assetData.data.asset_type} asset: {assetData.data.asset_name}");
                
                switch (assetData.data.asset_type.ToLower())
                {
                    case "texture":
                        ProcessTextureAsset(assetData.data);
                        break;
                        
                    case "material":
                        ProcessMaterialAsset(assetData.data);
                        break;
                        
                    default:
                        Debug.LogWarning($"Unknown asset type: {assetData.data.asset_type}");
                        break;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Process asset update error: {e.Message}");
        }
    }
    
    void ProcessTextureAsset(AssetData assetData)
    {
        try
        {
            // Decode base64 texture data
            byte[] textureBytes = Convert.FromBase64String(assetData.asset_data);
            
            // Load into asset system
            assetLoader.LoadAssetFromData("texture", assetData.asset_name, textureBytes);
            
            Debug.Log($"Loaded texture: {assetData.asset_name} ({textureBytes.Length} bytes)");
            
            // Send confirmation
            SendMessageToPython("asset_loaded", 
                $"Texture {assetData.asset_name} loaded successfully");
        }
        catch (Exception e)
        {
            Debug.LogError($"Process texture error: {e.Message}");
        }
    }
    
    void ProcessMaterialAsset(AssetData assetData)
    {
        try
        {
            // Material data is JSON
            byte[] materialBytes = Encoding.UTF8.GetBytes(assetData.asset_data);
            
            // Load into asset system
            assetLoader.LoadAssetFromData("material", assetData.asset_name, materialBytes);
            
            Debug.Log($"Loaded material: {assetData.asset_name}");
            
            // Send confirmation
            SendMessageToPython("asset_loaded", 
                $"Material {assetData.asset_name} loaded successfully");
        }
        catch (Exception e)
        {
            Debug.LogError($"Process material error: {e.Message}");
        }
    }
    
    void OnDestroy()
    {
        connected = false;
        if (receiveThread != null)
            receiveThread.Join(1000);
        if (stream != null)
            stream.Close();
        if (client != null)
            client.Close();
    }
}