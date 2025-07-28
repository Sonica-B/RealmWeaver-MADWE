using UnityEngine;

// Simple LOD component for tiles
public class TileLOD : MonoBehaviour
{
    private MeshRenderer[] renderers;
    private Collider[] colliders;
    private int currentLOD = 0;
    
    void Start()
    {
        renderers = GetComponentsInChildren<MeshRenderer>();
        colliders = GetComponentsInChildren<Collider>();
    }
    
    public void SetLOD(int lod)
    {
        if (currentLOD == lod) return;
        currentLOD = lod;
        
        switch (lod)
        {
            case 0: // Full detail
                SetRenderersEnabled(true);
                SetCollidersEnabled(true);
                break;
            case 1: // Medium detail
                SetRenderersEnabled(true);
                SetCollidersEnabled(false);
                break;
            case 2: // Low detail
                SetRenderersEnabled(true);
                SetCollidersEnabled(false);
                // Could swap to simpler mesh here
                break;
            case 3: // Very low detail
                SetRenderersEnabled(false);
                SetCollidersEnabled(false);
                break;
        }
    }
    
    void SetRenderersEnabled(bool enabled)
    {
        foreach (var renderer in renderers)
        {
            if (renderer != null)
                renderer.enabled = enabled;
        }
    }
    
    void SetCollidersEnabled(bool enabled)
    {
        foreach (var collider in colliders)
        {
            if (collider != null)
                collider.enabled = enabled;
        }
    }
}