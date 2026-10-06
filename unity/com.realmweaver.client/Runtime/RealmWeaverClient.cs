// HTTP side of the bridge (ADR-0003). DTOs are flat on purpose: JsonUtility reads neither dictionaries nor jagged
// arrays, so the bridge emits classes/tilesFlat/assetList/prefabList next to its tiles/assets/prefabs keys.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Networking;

namespace RealmWeaver.Client
{
    /// <summary>One key/value pair: the JsonUtility-readable form of a JSON object used as a map.</summary>
    [Serializable]
    public class KV
    {
        public string k;
        public string v;
    }

    /// <summary>GET /chunk/{cx}/{cy}. Field names are the JSON keys; tests/test_unity_protocol.py checks them.</summary>
    [Serializable]
    public class ChunkDto
    {
        public int cx;
        public int cy;
        public int size;            // tiles per side
        public string biome;
        public string state;        // pending | draft | ready
        public string[] classes;    // tile class names; tilesFlat indexes into this
        public int[] tilesFlat;     // size*size entries, row-major, row 0 first
        public List<KV> assetList;  // tile class -> asset id, fetched as /asset/<id>.png
        public List<KV> prefabList; // tile class -> prefab name, "Prefab_<tileClass>"
    }

    /// <summary>WS /events message. Only type == "ready" is acted on; "hello" parses with chunk == null.</summary>
    [Serializable]
    public class ReadyEvent
    {
        public string type;
        public int[] chunk;         // [cx, cy]
        public List<KV> assetList;  // tile class -> asset id (flat twin of the bridge's assets object)
    }

    /// <summary>Talks HTTP to the Python bridge (`uv run realmweaver serve`). One per scene; the others hold a reference.</summary>
    public sealed class RealmWeaverClient : MonoBehaviour
    {
        [SerializeField] string baseUrl = "http://127.0.0.1:8008";
        [SerializeField] string tier = "draft"; // quality tier asked of /chunk: draft | refine

        public string BaseUrl { get => baseUrl.TrimEnd('/'); set => baseUrl = value; }
        public string ChunkUrl(int cx, int cy) => $"{BaseUrl}/chunk/{cx}/{cy}?tier={tier}";
        public string AssetUrl(string assetId) => $"{BaseUrl}/asset/{assetId}.png";
        public string PlayerUrl => $"{BaseUrl}/player";

        /// <summary>ws(s)://host:port/events derived from BaseUrl (http -> ws, https -> wss).</summary>
        public string EventsUrl => Regex.Replace(BaseUrl, "^http", "ws", RegexOptions.IgnoreCase) + "/events";

        public void GetChunk(int cx, int cy, Action<ChunkDto> onDone, Action<string> onError)
            => StartCoroutine(GetChunkCo(cx, cy, onDone, onError));

        public void GetTexture(string assetId, Action<Texture2D> onDone, Action<string> onError)
            => StartCoroutine(GetTextureCo(assetId, onDone, onError));

        /// <summary>POST /player {"x","y"} in tile units; the bridge prewarms the chunks it predicts next.</summary>
        public void PostPlayer(float x, float y, Action<string> onError = null)
            => StartCoroutine(PostPlayerCo(x, y, onError));

        IEnumerator GetChunkCo(int cx, int cy, Action<ChunkDto> onDone, Action<string> onError)
        {
            using (var www = UnityWebRequest.Get(ChunkUrl(cx, cy)))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.Success) { onError?.Invoke(Describe(www)); yield break; }
                ChunkDto dto = null;
                string problem = null;
                try { dto = JsonUtility.FromJson<ChunkDto>(www.downloadHandler.text); }
                catch (ArgumentException e) { problem = e.Message; }
                if (problem == null && (dto == null || dto.classes == null || dto.tilesFlat == null
                                        || dto.tilesFlat.Length != dto.size * dto.size))
                    problem = "chunk JSON lacks the Unity keys (classes/tilesFlat) or tilesFlat.Length != size*size";
                if (problem != null) onError?.Invoke($"GET {www.url}: {problem}");
                else onDone?.Invoke(dto);
            }
        }

        IEnumerator GetTextureCo(string assetId, Action<Texture2D> onDone, Action<string> onError)
        {
            // nonReadable: the PNG is decoded off the main thread and no CPU copy is kept.
            using (var www = UnityWebRequestTexture.GetTexture(AssetUrl(assetId), true))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.Success) { onError?.Invoke(Describe(www)); yield break; }
                var tex = DownloadHandlerTexture.GetContent(www);
                tex.name = assetId;
                tex.wrapMode = TextureWrapMode.Repeat; // textures are seamless by construction
                tex.filterMode = FilterMode.Bilinear;
                onDone?.Invoke(tex);
            }
        }

        IEnumerator PostPlayerCo(float x, float y, Action<string> onError)
        {
            string body = string.Format(CultureInfo.InvariantCulture, "{{\"x\":{0},\"y\":{1}}}", x, y);
            using (var www = UnityWebRequest.Post(PlayerUrl, body, "application/json"))
            {
                yield return www.SendWebRequest();
                if (www.result != UnityWebRequest.Result.Success) onError?.Invoke(Describe(www));
            }
        }

        static string Describe(UnityWebRequest www) => $"{www.method} {www.url}: {www.error} (HTTP {www.responseCode})";
    }
}
