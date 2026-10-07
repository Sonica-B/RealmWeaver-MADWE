# ADR-0004: The world state graph is a networkx typed property graph and the single source of truth

Date: 2026-10-06. Status: accepted.

## Context
Coherence across chunks needs one place that knows every region, chunk, tile and asset plus their style. No paper defines a "neural world state graph"; closest prior art (AriGraph, StatePlay, MultiGen) agrees that a validated graph, not model context, must hold state.

## Decision
`networkx.MultiDiGraph` with node types World/Region/Chunk/Tile/Asset and edges CONTAINS, ADJACENT, INSTANCE_OF, STYLE_ANCHOR. One embedding type: DINOv2-small style vectors per asset, EMA style vector per region. Coherence = 0.6·cos(asset, region) + 0.4·mean cos(asset, adjacent); below threshold → regenerate twice → anchor fallback. Agents write only validated records. Persisted as node-link JSON.

## Consequences
No database. Graph size is bounded by the byte-capped cache. The embedder is injected so tests run on CPU with a histogram embedder.

## Amendment (2026-10-06, architecture review C4)
"Validated records" is delivered by one record schema (`realmweaver/world/records.py`): node kinds with their attributes and edge kinds with their allowed pairs, declared once. `add`/`link` validate on write; `validate`, `to_json` and `from_json` derive from the same tables. The game's kinds (Region3D, Settlement, Landmark, NPC, Memory, Quest, Objective, Faction, Event, Save; edges KNOWS, ASSIGNED, MEMBER_OF, TRIGGERS) are rows in those tables, not new graph methods.
