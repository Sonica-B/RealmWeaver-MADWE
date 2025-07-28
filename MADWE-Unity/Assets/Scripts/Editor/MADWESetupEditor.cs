using UnityEngine;
using UnityEditor;
using System.IO;

public class MADWESetupEditor : EditorWindow
{
    [MenuItem("MADWE/Setup Project Structure")]
    public static void ShowWindow()
    {
        GetWindow<MADWESetupEditor>("MADWE Setup");
    }
    
    void OnGUI()
    {
        GUILayout.Label("MADWE Project Setup", EditorStyles.boldLabel);
        
        EditorGUILayout.Space();
        
        if (GUILayout.Button("Create Folder Structure", GUILayout.Height(30)))
        {
            CreateFolderStructure();
        }
        
        EditorGUILayout.Space();
        
        if (GUILayout.Button("Create Default Prefabs", GUILayout.Height(30)))
        {
            CreateDefaultPrefabs();
        }
        
        EditorGUILayout.Space();
        
        if (GUILayout.Button("Create Default Materials", GUILayout.Height(30)))
        {
            CreateDefaultMaterials();
        }
        
        EditorGUILayout.Space();
        
        EditorGUILayout.HelpBox(
            "This will create the required folder structure in Assets/Resources for MADWE tile prefabs.",
            MessageType.Info
        );
    }
    
    void CreateFolderStructure()
    {
        // Create Resources folder
        CreateFolder("Assets", "Resources");
        
        // Create Prefabs structure
        CreateFolder("Assets/Resources", "Prefabs");
        CreateFolder("Assets/Resources/Prefabs", "Tiles");
        CreateFolder("Assets/Resources/Prefabs/Tiles", "Forest");
        CreateFolder("Assets/Resources/Prefabs/Tiles", "Desert");
        CreateFolder("Assets/Resources/Prefabs/Tiles", "Snow");
        CreateFolder("Assets/Resources/Prefabs", "Default");
        
        // Create Materials folder
        CreateFolder("Assets/Resources", "Materials");
        
        // Create StreamingAssets structure
        CreateFolder("Assets", "StreamingAssets");
        CreateFolder("Assets/StreamingAssets", "Textures");
        CreateFolder("Assets/StreamingAssets/Textures", "forest");
        CreateFolder("Assets/StreamingAssets/Textures", "desert");
        CreateFolder("Assets/StreamingAssets/Textures", "snow");
        CreateFolder("Assets/StreamingAssets", "Materials");
        
        AssetDatabase.Refresh();
        Debug.Log("MADWE folder structure created successfully!");
    }
    
    void CreateFolder(string parent, string folderName)
    {
        string path = parent + "/" + folderName;
        if (!AssetDatabase.IsValidFolder(path))
        {
            AssetDatabase.CreateFolder(parent, folderName);
        }
    }
    
    void CreateDefaultPrefabs()
    {
        // Forest prefabs
        CreateTilePrefab("Forest", "grass", PrefabType.Grass);
        CreateTilePrefab("Forest", "tree", PrefabType.Tree);
        CreateTilePrefab("Forest", "path", PrefabType.Path);
        CreateTilePrefab("Forest", "water", PrefabType.Water);
        
        // Desert prefabs
        CreateTilePrefab("Desert", "sand", PrefabType.Sand);
        CreateTilePrefab("Desert", "dune", PrefabType.Dune);
        CreateTilePrefab("Desert", "rock", PrefabType.Rock);
        CreateTilePrefab("Desert", "oasis", PrefabType.Water);
        
        // Snow prefabs
        CreateTilePrefab("Snow", "snow", PrefabType.Snow);
        CreateTilePrefab("Snow", "ice", PrefabType.Ice);
        CreateTilePrefab("Snow", "pine", PrefabType.Pine);
        
        // Default prefab
        CreateDefaultTilePrefab();
        
        AssetDatabase.Refresh();
        Debug.Log("Default prefabs created!");
    }
    
    enum PrefabType
    {
        Grass, Tree, Path, Water, Sand, Dune, Rock, Snow, Ice, Pine, Oasis
    }
    
    void CreateTilePrefab(string biome, string tileName, PrefabType type)
    {
        string path = $"Assets/Resources/Prefabs/Tiles/{biome}/{tileName}.prefab";
        
        // Check if already exists
        if (File.Exists(path))
        {
            Debug.Log($"Prefab already exists: {path}");
            return;
        }
        
        GameObject prefab = new GameObject(tileName);
        
        switch (type)
        {
            case PrefabType.Grass:
            case PrefabType.Sand:
            case PrefabType.Snow:
                CreateBasicTile(prefab, 0.5f);
                break;
                
            case PrefabType.Tree:
            case PrefabType.Pine:
                CreateTreeTile(prefab, type == PrefabType.Pine);
                break;
                
            case PrefabType.Path:
                CreateBasicTile(prefab, 0.2f);
                break;
                
            case PrefabType.Water:
            case PrefabType.Oasis:
                CreateWaterTile(prefab);
                break;
                
            case PrefabType.Dune:
                CreateDuneTile(prefab);
                break;
                
            case PrefabType.Rock:
                CreateRockTile(prefab);
                break;
                
            case PrefabType.Ice:
                CreateBasicTile(prefab, 0.3f);
                break;
        }
        
        // Add TileLOD component
        prefab.AddComponent<TileLOD>();
        
        // Save as prefab
        PrefabUtility.SaveAsPrefabAsset(prefab, path);
        DestroyImmediate(prefab);
    }
    
    void CreateBasicTile(GameObject parent, float height)
    {
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.transform.parent = parent.transform;
        cube.transform.localPosition = Vector3.zero;
        cube.transform.localScale = new Vector3(0.95f, height, 0.95f);
        cube.name = "Base";
    }
    
    void CreateTreeTile(GameObject parent, bool isPine)
    {
        // Trunk
        GameObject trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        trunk.transform.parent = parent.transform;
        trunk.transform.localPosition = Vector3.zero;
        trunk.transform.localScale = new Vector3(0.3f, isPine ? 1.5f : 1f, 0.3f);
        trunk.name = "Trunk";
        
        // Leaves
        if (isPine)
        {
            // Create cone-shaped leaves for pine
            for (int i = 0; i < 3; i++)
            {
                GameObject leaves = GameObject.CreatePrimitive(PrimitiveType.Cube);
                leaves.transform.parent = parent.transform;
                leaves.transform.localPosition = new Vector3(0, 1f + i * 0.5f, 0);
                leaves.transform.localScale = new Vector3(1.5f - i * 0.4f, 0.5f, 1.5f - i * 0.4f);
                leaves.transform.rotation = Quaternion.Euler(0, 45, 0);
                leaves.name = $"Leaves_{i}";
            }
        }
        else
        {
            // Regular tree leaves
            GameObject leaves = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            leaves.transform.parent = parent.transform;
            leaves.transform.localPosition = new Vector3(0, 1.5f, 0);
            leaves.transform.localScale = new Vector3(1.5f, 1.5f, 1.5f);
            leaves.name = "Leaves";
        }
    }
    
    void CreateWaterTile(GameObject parent)
    {
        GameObject water = GameObject.CreatePrimitive(PrimitiveType.Cube);
        water.transform.parent = parent.transform;
        water.transform.localPosition = new Vector3(0, -0.1f, 0);
        water.transform.localScale = new Vector3(0.95f, 0.3f, 0.95f);
        water.name = "Water";
    }
    
    void CreateDuneTile(GameObject parent)
    {
        GameObject dune = GameObject.CreatePrimitive(PrimitiveType.Cube);
        dune.transform.parent = parent.transform;
        dune.transform.localPosition = Vector3.zero;
        dune.transform.localScale = new Vector3(0.95f, 0.7f, 0.95f);
        dune.transform.rotation = Quaternion.Euler(0, Random.Range(-10f, 10f), 0);
        dune.name = "Dune";
    }
    
    void CreateRockTile(GameObject parent)
    {
        GameObject rock = GameObject.CreatePrimitive(PrimitiveType.Cube);
        rock.transform.parent = parent.transform;
        rock.transform.localPosition = new Vector3(0, 0.3f, 0);
        rock.transform.localScale = new Vector3(0.7f, 0.8f, 0.6f);
        rock.transform.rotation = Quaternion.Euler(
            Random.Range(-15f, 15f),
            Random.Range(0f, 360f),
            Random.Range(-15f, 15f)
        );
        rock.name = "Rock";
    }
    
    void CreateDefaultTilePrefab()
    {
        string path = "Assets/Resources/Prefabs/Default/Default_Tile.prefab";
        
        if (File.Exists(path))
        {
            Debug.Log("Default tile already exists");
            return;
        }
        
        GameObject prefab = new GameObject("Default_Tile");
        CreateBasicTile(prefab, 0.5f);
        prefab.AddComponent<TileLOD>();
        
        PrefabUtility.SaveAsPrefabAsset(prefab, path);
        DestroyImmediate(prefab);
    }
    
    void CreateDefaultMaterials()
    {
        CreateMaterial("Forest_Mat", new Color(0.2f, 0.5f, 0.2f), 0f, 0.2f);
        CreateMaterial("Desert_Mat", new Color(0.9f, 0.8f, 0.6f), 0f, 0.1f);
        CreateMaterial("Snow_Mat", new Color(0.95f, 0.95f, 1.0f), 0.1f, 0.7f);
        
        AssetDatabase.Refresh();
        Debug.Log("Default materials created!");
    }
    
    void CreateMaterial(string name, Color color, float metallic, float smoothness)
    {
        string path = $"Assets/Resources/Materials/{name}.mat";
        
        if (File.Exists(path))
        {
            Debug.Log($"Material already exists: {name}");
            return;
        }
        
        Material mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        mat.SetFloat("_Metallic", metallic);
        mat.SetFloat("_Glossiness", smoothness);
        
        AssetDatabase.CreateAsset(mat, path);
    }
}