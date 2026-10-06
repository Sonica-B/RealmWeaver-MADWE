# ADR-0002: Own numpy WFC solver, rules learned from an ASCII example map

Date: 2026-10-06. Status: accepted.

## Context
The 2025 configs were 62k lines of fully connected adjacency over tiles that no longer exist, so WFC degenerated to uniform random tiling. No viable PyPI WFC package exists (`wfc` is a web framework, `pywfc` a 2021 v0.0.0 toy).

## Decision
`realmweaver.layout` implements the simple tiled model in ~200 lines: directional `allowed[T,4,T]` and weights learned from each biome's `map.txt`, min-entropy observation, AC-3 propagation with boolean matrix ops, restart on contradiction (≤5 seeds) then shrink. Chunks solve with fixed border cells copied from ready neighbours.

## Consequences
Rules are authored as a 12×12 ASCII map anyone can edit; tests verify zero adjacency violations with an independent checker. Hierarchical/meta-tile WFC and path/symmetry constraints are out of scope.
