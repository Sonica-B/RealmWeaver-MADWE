# Coding Standards

1. Vocabulary from `GLOSSARY.md`: biome, tile class, tileset, chunk, asset, asset spec, generator, world state graph, predictor, scheduler, bridge. No synonyms in code or docs.
2. Deep modules, few seams. Public interface of each package is listed in its `__init__.py`; everything else is private. The only cross-package seams are `Generator.generate`, `layout.solve`, `World`, and the bridge routes.
3. Accept dependencies, return results. No module constructs a model or opens a file a caller could inject; pure functions return values instead of mutating inputs.
4. Dataclasses over dicts for anything that crosses a seam; `numpy.ndarray` uint8 HxWx3|4 for images, PIL only at file/HTTP edges.
5. No hard-coded machine paths, no secrets, no `print` in library code (use `logging`). Settings come from `realmweaver.config`.
6. GPU code is optional: every public path has a CPU fallback or a clear `RuntimeError` naming the missing requirement.
7. Tests live at the seams, use independent expected values, never mock internals. CPU tests must run under 60 s total; GPU tests carry `@pytest.mark.gpu`.
8. Deliberate shortcuts carry a `# ponytail:` comment naming the ceiling and the upgrade path.
9. ruff (`ruff check` + `ruff format`) is the only linter/formatter; type hints on every public function.
10. Performance and quality claims come from a benchmark report; no literal performance numbers in code, docs or UI. Live operational state (chunk counts, cache bytes, queue lengths) may be shown when it is labelled as state, never presented as a measurement.
