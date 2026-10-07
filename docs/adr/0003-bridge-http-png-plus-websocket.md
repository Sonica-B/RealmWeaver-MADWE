# ADR-0003: Bridge is HTTP PNG + chunk JSON with a WebSocket event channel; diffusion stays in Python

Date: 2026-10-06. Status: accepted.

## Context
The repo had a 55-line TCP stub with no framing (`recv(4096)`). Unity Sentis cannot run an SD UNet usefully. Unity's own texture path (`UnityWebRequestTexture`, worker-thread decode) is the cheapest robust transport.

## Decision
FastAPI serves `GET /chunk/{cx}/{cy}` (JSON: tile classes, asset ids, prefab map), `GET /asset/{id}.png`, `POST /generate`, `POST /player`, `GET /report`, and `WS /events` for `ready` events. Unity side: `AssetStreamer` (priority queue, ≤4 coroutines, placeholder, byte-capped LRU) + `TileCatalog` ScriptableObject + NativeWebSocket client. Generation never runs inside the game process.

## Consequences
Any client that speaks HTTP can consume the world (the operator page uses the same endpoints). KTX2/Basis only if transfer is measured as the bottleneck.
