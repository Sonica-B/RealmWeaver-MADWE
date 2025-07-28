using UnityEngine;

public class SimpleTileTest : MonoBehaviour
{
    void Start()
    {
        // Test tile creation directly
        TileManager tileManager = GetComponent<TileManager>();
        if (tileManager != null)
        {
            int[,] testGrid = new int[,] {
                {0, 1, 0, 2, 0},
                {1, 0, 0, 0, 3},
                {0, 0, 2, 0, 0},
                {2, 0, 0, 1, 0},
                {0, 3, 0, 0, 0}
            };
            
            tileManager.CreateTileGrid(testGrid, "forest");
            Debug.Log("Test grid created!");
        }
    }
}