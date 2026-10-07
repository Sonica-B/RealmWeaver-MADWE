using System;
using System.Collections.Generic;
using UnityEngine;

namespace RealmWeaver.Client
{
    /// <summary>Prefab map: tile class -> prefab. Several entries with the same tile class are variants; Resolve picks
    /// one deterministically from the world tile position. Name prefabs Prefab_&lt;tileClass&gt; to match the bridge.</summary>
    [CreateAssetMenu(fileName = "TileCatalog", menuName = "RealmWeaver/Tile Catalog")]
    public sealed class TileCatalog : ScriptableObject
    {
        [Serializable]
        public class Entry
        {
            public string tileClass;
            public GameObject prefab;
        }

        public List<Entry> entries = new List<Entry>();
        [Tooltip("Used for tile classes without an entry; leave empty to skip such tiles.")]
        public GameObject fallback;

        Dictionary<string, List<GameObject>> index;

        public GameObject Resolve(string tileClass) => Resolve(tileClass, Vector2Int.zero);

        public GameObject Resolve(string tileClass, Vector2Int worldTile)
        {
            if (index == null) BuildIndex();
            if (string.IsNullOrEmpty(tileClass) || !index.TryGetValue(tileClass, out var variants)) return fallback;
            return variants.Count == 1 ? variants[0] : variants[VariantIndex(worldTile, variants.Count)];
        }

        /// <summary>Position hash (adapted from origin/unity GetVariationIndex) so a reloaded chunk picks the same variants.</summary>
        static int VariantIndex(Vector2Int p, int count)
        {
            int hash = p.x * 73856093 ^ p.y * 19349663;
            return (hash & 0x7fffffff) % count;
        }

        void BuildIndex()
        {
            index = new Dictionary<string, List<GameObject>>();
            foreach (var e in entries)
            {
                if (e == null || e.prefab == null || string.IsNullOrEmpty(e.tileClass)) continue;
                if (!index.TryGetValue(e.tileClass, out var list)) index[e.tileClass] = list = new List<GameObject>();
                list.Add(e.prefab);
            }
        }

        void OnEnable() => index = null;
        void OnValidate() => index = null; // Inspector edits rebuild the index lazily
    }
}
