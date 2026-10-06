# RealmWeaver Unity client (`com.realmweaver.client`)

Unity 6.3 LTS package that streams RealmWeaver chunks and generated textures from the Python bridge into a scene.
Generation never runs inside the game process (ADR-0003): Unity fetches chunk JSON and PNG textures over HTTP and hears
about finished assets over a WebSocket, or by polling when the WebSocket package is absent.

| Script | Role |
|---|---|
| `RealmWeaverClient` | Base URL; `GetChunk`, `GetTexture` (`UnityWebRequestTexture`, non-readable), `PostPlayer`; the JSON DTOs |
| `AssetStreamer` | Texture queue ordered by chunk distance to the player, at most 4 requests in flight, placeholder texture, byte-capped LRU that `Destroy`s evicted textures |
| `TileCatalog` | ScriptableObject prefab map, tile class -> prefab; several entries per class are variants picked by a position hash |
| `ChunkRenderer` | Keeps the chunks around the player loaded, one prefab per tile, textures applied through `MaterialPropertyBlock` (`_BaseMap` and `_MainTex`) |
| `PlayerReporter` | POSTs the player tile position to `/player` at most 4 times per second so the bridge can prewarm |
| `RealmWeaverEvents` | `ready` events: WebSocket `/events` via NativeWebSocket when `REALMWEAVER_WS` is defined, otherwise polls `/chunk` state every 2 s |

## Install

1. Unity 6.3 LTS (6000.3) or newer. URP or Built-in: the client sets both `_BaseMap` and `_MainTex`.
2. Window > Package Manager > `+` > **Add package from disk...** > select `unity/com.realmweaver.client/package.json`
   in this repository. Equivalent `Packages/manifest.json` line: `"com.realmweaver.client": "file:<path>/unity/com.realmweaver.client"`.
3. Optional WebSocket events: `+` > **Add package from git URL...** > `https://github.com/endel/NativeWebSocket.git#upm-2`
   (NativeWebSocket 2.x; `#upm` installs 1.x, also supported). The assembly definition defines `REALMWEAVER_WS` by itself
   while `com.endel.nativewebsocket` (any version from 1.0.0) is installed; to force the symbol instead, add
   `REALMWEAVER_WS` under Project Settings > Player > Scripting Define Symbols. Without the package the HTTP polling
   fallback compiles.

## Run the bridge

From the repository root:

    uv sync
    uv run realmweaver serve        # http://127.0.0.1:8008 ; --procedural skips the GPU generator

`GET /health` reports which generator is active. The operator page at `/` uses the same endpoints as this client.

## Scene setup

1. One GameObject `RealmWeaver` with, in order: `RealmWeaverClient` (Base Url `http://127.0.0.1:8008`, Tier `draft`),
   `AssetStreamer` (Client; Max In Flight 4; Max Cache Bytes), `ChunkRenderer` (Client, Streamer, Catalog, Player,
   Tile Size, View Radius), `RealmWeaverEvents` (Client, Chunks) and `PlayerReporter` (Client, Chunks, Player).
2. Tile prefabs: one per tile class, named `Prefab_<tileClass>` (`Prefab_grass`, `Prefab_water`, ...), one tile unit
   wide, carrying a `Renderer` (a quad `MeshRenderer` with a URP Lit/Unlit or Built-in material, or a `SpriteRenderer`).
   Textures are applied per instance through a `MaterialPropertyBlock`, so every prefab can share one material.
3. Assets > Create > RealmWeaver > Tile Catalog; one entry per tile class (`tileClass` = the name the bridge sends,
   `prefab` = the prefab). Several entries with the same class are variants chosen deterministically per tile position;
   `fallback` covers classes without an entry (empty skips them). Assign the asset to `ChunkRenderer`.
4. A player `Transform` (any moving object or the camera). Tile `(x, y)` of chunk `(cx, cy)` sits at world
   `((cx*size + x) * tileSize, -(cy*size + y) * tileSize, 0)`: JSON row 0 is the top row and y grows downwards, so the
   scene reads like the JSON. `PlayerReporter` inverts the same mapping when it posts positions.

Play: chunks within `viewRadius` of the player's chunk appear with the grey placeholder, textures stream in nearest
chunk first, and `ready` events (or the poll) refresh chunks whose assets finished later.

## JSON contract

`GET /chunk/{cx}/{cy}?tier=draft|refine`; `tests/fixtures/chunk_example.json` is the reference payload.

| Key | Type | Unity |
|---|---|---|
| `cx`, `cy`, `size` | int | read |
| `biome`, `state` (`pending`, `draft`, `ready`) | string | read |
| `classes` | string[] | read: tile class names |
| `tilesFlat` | int[] of length `size*size` | read: row-major indices into `classes` |
| `assetList`, `prefabList` | `[{"k": tileClass, "v": assetId or "Prefab_<tileClass>"}]` | read |
| `tiles`, `assets`, `prefabs` | nested rows, objects | ignored: `JsonUtility` reads neither jagged arrays nor dictionaries; kept for other clients |

`GET /asset/{id}.png` returns the PNG (`wrapMode = Repeat`). `POST /player` takes `{"x": <float>, "y": <float>}` in tile
units. `WS /events` sends `{"type": "hello"}` on connect, then `{"type": "ready", "chunk": [cx, cy], "assets": {cls: id}, "assetList": [{"k", "v"}]}`;
the client acts on `type` and `chunk` only and re-fetches the chunk.

`tests/test_unity_protocol.py` parses the `[Serializable]` DTOs in `RealmWeaverClient.cs` and asserts every field is a
key of the fixture; the bridge test asserts its live response carries the fixture keys. Unity itself is not compiled in CI.
