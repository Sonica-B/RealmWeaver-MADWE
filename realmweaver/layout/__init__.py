"""Layout agent: tilesets learned from example maps, the numpy WFC solver, chunk borders and a preview renderer."""

from realmweaver.layout.render import render
from realmweaver.layout.tileset import tileset_from_example
from realmweaver.layout.wfc import Contradiction, solve, solve_chunk, violations

__all__ = ["Contradiction", "render", "solve", "solve_chunk", "tileset_from_example", "violations"]
