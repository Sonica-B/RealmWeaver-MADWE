using System.Collections.Generic;
using UnityEngine;

namespace RealmWeaver.Client
{
    /// <summary>Keeps the chunks around the player loaded: one GameObject per chunk, one TileCatalog prefab per tile,
    /// textures applied through a MaterialPropertyBlock (_BaseMap for URP, _MainTex for Built-in and sprites).
    /// Tile (x, y) of chunk (cx, cy) sits at ((cx*size + x) * tileSize, -(cy*size + y) * tileSize, 0): row 0 is the top.</summary>
    public sealed class ChunkRenderer : MonoBehaviour
    {
        sealed class View
        {
            public GameObject Root;
            public ChunkDto Dto;
            public readonly Dictionary<string, List<Renderer>> ByClass = new Dictionary<string, List<Renderer>>();
        }

        static readonly int BaseMap = Shader.PropertyToID("_BaseMap");
        static readonly int MainTex = Shader.PropertyToID("_MainTex");

        [SerializeField] RealmWeaverClient client;
        [SerializeField] AssetStreamer streamer;
        [SerializeField] TileCatalog catalog;
        [SerializeField] Transform player;
        [SerializeField, Min(0.01f)] float tileSize = 1f;
        [SerializeField, Min(0)] int viewRadius = 1; // chunks kept loaded around the chunk the player stands in (Chebyshev)
        [SerializeField, Min(1)] int chunkSize = 16; // bridge chunk size; overwritten by the first chunk received

        readonly Dictionary<Vector2Int, View> views = new Dictionary<Vector2Int, View>();
        readonly HashSet<Vector2Int> pending = new HashSet<Vector2Int>();
        readonly List<Vector2Int> farAway = new List<Vector2Int>();
        readonly MaterialPropertyBlock block = new MaterialPropertyBlock();

        public float TileSize => tileSize;
        public IEnumerable<ChunkDto> Loaded { get { foreach (var v in views.Values) yield return v.Dto; } }

        public Vector2Int WorldToTile(Vector3 world)
            => new Vector2Int(Mathf.FloorToInt(world.x / tileSize), Mathf.FloorToInt(-world.y / tileSize));

        public Vector2Int ChunkOf(Vector2Int tile)
            => new Vector2Int(Mathf.FloorToInt((float)tile.x / chunkSize), Mathf.FloorToInt((float)tile.y / chunkSize));

        void Update()
        {
            if (player == null || client == null) return;
            var centre = ChunkOf(WorldToTile(player.position));
            if (streamer != null) streamer.PlayerChunk = centre;
            for (int dy = -viewRadius; dy <= viewRadius; dy++)
                for (int dx = -viewRadius; dx <= viewRadius; dx++)
                    Load(centre.x + dx, centre.y + dy);
            farAway.Clear();
            foreach (var key in views.Keys)
                if (Mathf.Max(Mathf.Abs(key.x - centre.x), Mathf.Abs(key.y - centre.y)) > viewRadius + 1) farAway.Add(key);
            foreach (var key in farAway) Unload(key);
        }

        /// <summary>Fetches a chunk once; no-op while it is loaded or in flight.</summary>
        public void Load(int cx, int cy)
        {
            var key = new Vector2Int(cx, cy);
            if (views.ContainsKey(key) || !pending.Add(key)) return;
            client.GetChunk(cx, cy,
                dto => { pending.Remove(key); Show(dto); },
                err => { pending.Remove(key); Debug.LogWarning(err, this); });
        }

        /// <summary>Re-fetches a loaded chunk (after a ready event or a state poll) and re-applies its textures.</summary>
        public void Refresh(int cx, int cy)
        {
            var key = new Vector2Int(cx, cy);
            if (!views.ContainsKey(key)) return;
            client.GetChunk(cx, cy, dto => { if (views.ContainsKey(key)) Show(dto); }, err => Debug.LogWarning(err, this));
        }

        /// <summary>Builds the chunk tiles on first sight; later calls only re-apply textures.</summary>
        public void Show(ChunkDto dto)
        {
            var key = new Vector2Int(dto.cx, dto.cy);
            if (!views.TryGetValue(key, out var view)) views[key] = view = Build(dto);
            view.Dto = dto; // ponytail: a chunk layout is deterministic, so only state and asset ids change between fetches
            ApplyTextures(view);
        }

        View Build(ChunkDto dto)
        {
            chunkSize = dto.size;
            var view = new View { Root = new GameObject($"Chunk_{dto.cx}_{dto.cy}") };
            view.Root.transform.SetParent(transform, false);
            view.Root.transform.localPosition = TileToLocal(dto.cx * dto.size, dto.cy * dto.size);
            for (int i = 0; i < dto.tilesFlat.Length; i++)
            {
                int x = i % dto.size, y = i / dto.size, cls = dto.tilesFlat[i];
                if ((uint)cls >= (uint)dto.classes.Length) continue;
                string tileClass = dto.classes[cls];
                var prefab = catalog != null
                    ? catalog.Resolve(tileClass, new Vector2Int(dto.cx * dto.size + x, dto.cy * dto.size + y)) : null;
                if (prefab == null) continue;
                var tile = Instantiate(prefab, view.Root.transform);
                tile.name = $"{tileClass}_{x}_{y}";
                tile.transform.localPosition = TileToLocal(x, y);
                if (!view.ByClass.TryGetValue(tileClass, out var list)) view.ByClass[tileClass] = list = new List<Renderer>();
                list.AddRange(tile.GetComponentsInChildren<Renderer>());
            }
            return view;
        }

        void ApplyTextures(View view)
        {
            if (view.Dto.assetList == null || streamer == null) return;
            var key = new Vector2Int(view.Dto.cx, view.Dto.cy);
            foreach (var kv in view.Dto.assetList)
            {
                if (!view.ByClass.TryGetValue(kv.k, out var renderers)) continue;
                SetTexture(renderers, streamer.Request(kv.v, key, tex => SetTexture(renderers, tex)));
            }
        }

        void SetTexture(List<Renderer> renderers, Texture2D tex)
        {
            if (tex == null) return;
            foreach (var r in renderers)
            {
                if (r == null) continue; // tile destroyed while the request was in flight
                r.GetPropertyBlock(block);
                block.SetTexture(BaseMap, tex);
                block.SetTexture(MainTex, tex);
                r.SetPropertyBlock(block);
            }
        }

        void Unload(Vector2Int key)
        {
            if (!views.TryGetValue(key, out var view)) return;
            views.Remove(key);
            Destroy(view.Root);
        }

        Vector3 TileToLocal(int x, int y) => new Vector3(x * tileSize, -y * tileSize, 0f);
    }
}
