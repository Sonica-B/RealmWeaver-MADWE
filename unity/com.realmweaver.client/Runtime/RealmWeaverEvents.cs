// Ready events from the bridge. With REALMWEAVER_WS (set by the asmdef version define while com.endel.nativewebsocket
// is installed, or by hand in Scripting Define Symbols) this is WS /events; otherwise it polls GET /chunk/{cx}/{cy}
// state for every loaded chunk that is not ready yet.
using System.Collections;
using System.Text;
using UnityEngine;
#if REALMWEAVER_WS
using NativeWebSocket;
#endif

namespace RealmWeaver.Client
{
    /// <summary>Turns bridge "ready" events into ChunkRenderer.Refresh calls so placeholders get their textures.</summary>
    public sealed class RealmWeaverEvents : MonoBehaviour
    {
        [SerializeField] RealmWeaverClient client;
        [SerializeField] ChunkRenderer chunks;
        [SerializeField, Min(0.2f)] float pollSeconds = 2f;      // HTTP fallback only
        [SerializeField, Min(0.5f)] float reconnectSeconds = 3f; // WebSocket only

        /// <summary>Raised per ready event in WebSocket mode; the poller refreshes chunks directly.</summary>
        public event System.Action<ReadyEvent> Ready;

        void OnMessage(string json)
        {
            ReadyEvent ev;
            try { ev = JsonUtility.FromJson<ReadyEvent>(json); }
            catch (System.ArgumentException e) { Debug.LogWarning($"events: bad JSON ({e.Message})", this); return; }
            if (ev == null || ev.type != "ready" || ev.chunk == null || ev.chunk.Length != 2) return;
            Ready?.Invoke(ev);
            if (chunks != null) chunks.Refresh(ev.chunk[0], ev.chunk[1]);
        }

#if REALMWEAVER_WS
        WebSocket socket;

        void Start() => StartCoroutine(Run());

        IEnumerator Run()
        {
            while (enabled)
            {
                bool closed = false;
                socket = new WebSocket(client.EventsUrl);
                socket.OnMessage += bytes => OnMessage(Encoding.UTF8.GetString(bytes));
                socket.OnError += err => Debug.LogWarning($"events socket: {err}", this);
                socket.OnClose += code => closed = true;
                var session = socket.Connect(); // NativeWebSocket 1.x and 2.x: the task completes when the socket closes
                while (!closed && !session.IsCompleted) yield return null;
                socket = null;
                yield return new WaitForSeconds(reconnectSeconds);
            }
        }

        void Update()
        {
#if !UNITY_WEBGL || UNITY_EDITOR
            socket?.DispatchMessageQueue(); // 1.x raises OnMessage on the main thread here; a no-op drain on 2.x
#endif
        }

        void OnDestroy()
        {
            var s = socket;
            socket = null;
            if (s != null && s.State == WebSocketState.Open) _ = s.Close();
        }
#else
        void Start() => StartCoroutine(Poll());

        // ponytail: refetches the whole chunk JSON for every not-ready chunk; a state-only route would be lighter.
        IEnumerator Poll()
        {
            while (enabled)
            {
                yield return new WaitForSeconds(pollSeconds);
                if (chunks == null) continue;
                foreach (var dto in chunks.Loaded)
                    if (dto.state != "ready") chunks.Refresh(dto.cx, dto.cy);
            }
        }
#endif
    }
}
