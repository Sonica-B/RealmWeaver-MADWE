// Adapted from origin/unity AssetLoadingSystem.cs (docs/research/03-origin-unity-salvage.md, section 2.6). Kept: the LRU
// with Destroy on evict, the in-flight counter that drops on success and on error, ClearCache/OnDestroy. Replaced: the
// StreamingAssets/material code with HTTP via RealmWeaverClient, a byte cap, nearest-chunk-first order, a placeholder.
using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Profiling;

namespace RealmWeaver.Client
{
    /// <summary>Streams asset textures: nearest chunk first, at most maxInFlight requests, byte-capped LRU cache.</summary>
    public sealed class AssetStreamer : MonoBehaviour
    {
        sealed class Job { public string Id; public Vector2Int Chunk; public Action<Texture2D> OnLoaded; }
        sealed class Entry { public string Id; public Texture2D Tex; public long Bytes; }

        [SerializeField] RealmWeaverClient client;
        [SerializeField, Range(1, 16)] int maxInFlight = 4;
        [SerializeField] long maxCacheBytes = 256L * 1024 * 1024;

        /// <summary>Chunk the player stands in (ChunkRenderer sets it); requests nearest to it go first.</summary>
        public Vector2Int PlayerChunk { get; set; }
        public int InFlight => inFlight;
        public int Queued => queue.Count;
        public long CachedBytes => cachedBytes;

        readonly Dictionary<string, LinkedListNode<Entry>> cache = new Dictionary<string, LinkedListNode<Entry>>();
        readonly LinkedList<Entry> lru = new LinkedList<Entry>(); // head = least recently used
        readonly Dictionary<string, Job> jobs = new Dictionary<string, Job>(); // queued or in flight; merges duplicates
        readonly List<Job> queue = new List<Job>(); // not yet dispatched
        int inFlight;
        long cachedBytes;
        Texture2D placeholder;

        /// <summary>2x2 grey checker shown until the real texture arrives.</summary>
        public Texture2D Placeholder { get { if (placeholder == null) placeholder = MakePlaceholder(); return placeholder; } }

        /// <summary>Returns what to show now (cached texture or placeholder); onLoaded fires when the fetch completes.</summary>
        public Texture2D Request(string assetId, Vector2Int chunk, Action<Texture2D> onLoaded)
        {
            if (string.IsNullOrEmpty(assetId)) return Placeholder;
            if (cache.TryGetValue(assetId, out var node))
            {
                if (node.Value.Tex != null) { lru.Remove(node); lru.AddLast(node); return node.Value.Tex; }
                Evict(node); // destroyed behind our back: fetch again
            }
            if (jobs.TryGetValue(assetId, out var job)) { job.Chunk = chunk; job.OnLoaded += onLoaded; }
            else { jobs[assetId] = job = new Job { Id = assetId, Chunk = chunk, OnLoaded = onLoaded }; queue.Add(job); }
            return Placeholder;
        }

        void Update()
        {
            if (client == null) return;
            while (inFlight < maxInFlight && queue.Count > 0) Dispatch(PopNearest());
        }

        // ponytail: O(n) scan per dispatch; fine for hundreds of jobs, and priorities shift as the player moves anyway.
        Job PopNearest()
        {
            int best = 0, bestDist = int.MaxValue;
            for (int i = 0; i < queue.Count; i++)
            {
                var c = queue[i].Chunk;
                int d = Mathf.Max(Mathf.Abs(c.x - PlayerChunk.x), Mathf.Abs(c.y - PlayerChunk.y));
                if (d < bestDist) { bestDist = d; best = i; }
            }
            var job = queue[best];
            queue.RemoveAt(best);
            return job;
        }

        void Dispatch(Job job)
        {
            inFlight++;
            client.GetTexture(job.Id,
                tex => { Finish(job); Insert(job.Id, tex); job.OnLoaded?.Invoke(tex); },
                err => { Finish(job); Debug.LogWarning($"asset {job.Id}: {err}", this); }); // a ready event or poll retries
        }

        void Finish(Job job) { inFlight--; jobs.Remove(job.Id); }

        void Insert(string id, Texture2D tex)
        {
            if (cache.TryGetValue(id, out var old)) Evict(old);
            long bytes = Profiler.GetRuntimeMemorySizeLong(tex);
            if (bytes <= 0) bytes = (long)tex.width * tex.height * 4; // profiler unavailable in release players
            cache[id] = lru.AddLast(new Entry { Id = id, Tex = tex, Bytes = bytes });
            cachedBytes += bytes;
            // ponytail: strict LRU. A cap below the visible working set evicts textures still on screen (they show the
            // shader default until the chunk refreshes); raise maxCacheBytes rather than pin.
            while (cachedBytes > maxCacheBytes && lru.Count > 1) Evict(lru.First);
        }

        void Evict(LinkedListNode<Entry> node)
        {
            lru.Remove(node);
            cache.Remove(node.Value.Id);
            cachedBytes -= node.Value.Bytes;
            if (node.Value.Tex != null) Destroy(node.Value.Tex);
        }

        public void ClearCache() { while (lru.Count > 0) Evict(lru.First); }

        void OnDestroy()
        {
            ClearCache();
            if (placeholder != null) Destroy(placeholder);
        }

        static Texture2D MakePlaceholder()
        {
            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false)
                { name = "RealmWeaver placeholder", wrapMode = TextureWrapMode.Repeat, filterMode = FilterMode.Point };
            Color32 dark = new Color32(88, 88, 88, 255), light = new Color32(136, 136, 136, 255);
            tex.SetPixels32(new[] { dark, light, light, dark });
            tex.Apply(false, true);
            return tex;
        }
    }
}
