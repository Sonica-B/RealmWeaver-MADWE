using UnityEngine;

namespace RealmWeaver.Client
{
    /// <summary>POSTs the player tile position to /player at most hz times per second so the bridge can prewarm.
    /// Own file because Unity only attaches MonoBehaviours whose file name matches the class name.</summary>
    public sealed class PlayerReporter : MonoBehaviour
    {
        [SerializeField] RealmWeaverClient client;
        [SerializeField] ChunkRenderer chunks; // supplies the world-to-tile convention
        [SerializeField] Transform player;
        [SerializeField, Range(0.1f, 4f)] float hz = 4f;
        [SerializeField] bool skipWhenStill = true;

        float nextSend;
        Vector2 lastSent = new Vector2(float.NaN, float.NaN);

        void Update()
        {
            if (client == null || player == null || Time.unscaledTime < nextSend) return;
            nextSend = Time.unscaledTime + 1f / hz;
            float size = chunks != null ? chunks.TileSize : 1f;
            var pos = new Vector2(player.position.x / size, -player.position.y / size);
            if (skipWhenStill && pos == lastSent) return;
            lastSent = pos;
            client.PostPlayer(pos.x, pos.y, err => Debug.LogWarning(err, this));
        }
    }
}
