"""
Hierarchical Wave Function Collapse Implementation
Day 3: Core WFC with neural constraints
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import json
from pathlib import Path
import random
from collections import defaultdict
import torch
import torch.nn as nn


@dataclass
class Tile:
    """Represents a tile in the WFC system"""
    id: int
    name: str
    biome: str
    rotation: int = 0
    weight: float = 1.0
    edges: Dict[str, str] = None  # N, E, S, W edge types
    
    def __post_init__(self):
        if self.edges is None:
            self.edges = {'N': 'default', 'E': 'default', 'S': 'default', 'W': 'default'}


class WaveFunctionCollapse:
    """Basic WFC implementation"""
    
    def __init__(self, tiles: List[Tile], grid_size: Tuple[int, int]):
        self.tiles = {tile.id: tile for tile in tiles}
        self.grid_size = grid_size
        self.width, self.height = grid_size
        
        # Wave function: each cell contains possible tile IDs
        self.wave = [[set(self.tiles.keys()) for _ in range(self.width)] 
                     for _ in range(self.height)]
        
        # Collapsed state: final tile assignments
        self.collapsed = [[None for _ in range(self.width)] 
                          for _ in range(self.height)]
        
        # Adjacency rules
        self.adjacency_rules = self._compute_adjacency_rules()
        
        # Propagation stack
        self.propagation_stack = []
        
    def _compute_adjacency_rules(self) -> Dict[str, Dict[str, Set[int]]]:
        """Compute which tiles can be adjacent based on edge compatibility"""
        rules = {
            'N': defaultdict(set),
            'E': defaultdict(set),
            'S': defaultdict(set),
            'W': defaultdict(set)
        }
        
        opposite = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}
        
        for tile_id, tile in self.tiles.items():
            for direction, edge in tile.edges.items():
                # Find compatible tiles
                for other_id, other_tile in self.tiles.items():
                    if other_tile.edges[opposite[direction]] == edge:
                        rules[direction][tile_id].add(other_id)
                        
        return rules
        
    def collapse(self) -> np.ndarray:
        """Run the WFC algorithm"""
        iteration = 0
        max_iterations = self.width * self.height * 10
        
        while not self._is_fully_collapsed() and iteration < max_iterations:
            # Find cell with minimum entropy (non-zero)
            cell = self._find_min_entropy_cell()
            
            if cell is None:
                # No valid cells, contradiction
                raise ValueError("WFC reached contradiction state")
                
            x, y = cell
            
            # Collapse the cell
            self._collapse_cell(x, y)
            
            # Propagate constraints
            self._propagate()
            
            iteration += 1
            
        if iteration >= max_iterations:
            raise ValueError("WFC exceeded maximum iterations")
            
        # Convert to numpy array
        result = np.zeros((self.height, self.width), dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                if self.collapsed[y][x] is not None:
                    result[y, x] = self.collapsed[y][x]
                    
        return result
        
    def _is_fully_collapsed(self) -> bool:
        """Check if all cells are collapsed"""
        for y in range(self.height):
            for x in range(self.width):
                if self.collapsed[y][x] is None:
                    return False
        return True
        
    def _find_min_entropy_cell(self) -> Optional[Tuple[int, int]]:
        """Find uncollapsed cell with minimum entropy"""
        min_entropy = float('inf')
        candidates = []
        
        for y in range(self.height):
            for x in range(self.width):
                if self.collapsed[y][x] is None:
                    entropy = len(self.wave[y][x])
                    
                    if entropy == 0:
                        continue  # Skip contradictions
                        
                    if entropy < min_entropy:
                        min_entropy = entropy
                        candidates = [(x, y)]
                    elif entropy == min_entropy:
                        candidates.append((x, y))
                        
        if not candidates:
            return None
            
        # Choose randomly among minimum entropy cells
        return random.choice(candidates)
        
    def _collapse_cell(self, x: int, y: int):
        """Collapse a cell to a single tile"""
        possible_tiles = list(self.wave[y][x])
        
        if not possible_tiles:
            raise ValueError(f"No possible tiles for cell ({x}, {y})")
            
        # Weight-based selection
        weights = [self.tiles[tile_id].weight for tile_id in possible_tiles]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        chosen_tile = np.random.choice(possible_tiles, p=weights)
        
        # Collapse the cell
        self.collapsed[y][x] = chosen_tile
        self.wave[y][x] = {chosen_tile}
        
        # Add neighbors to propagation stack
        self._add_neighbors_to_stack(x, y)
        
    def _add_neighbors_to_stack(self, x: int, y: int):
        """Add neighboring cells to propagation stack"""
        neighbors = [
            (x, y-1, 'N'),  # North
            (x+1, y, 'E'),  # East
            (x, y+1, 'S'),  # South
            (x-1, y, 'W')   # West
        ]
        
        for nx, ny, direction in neighbors:
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.collapsed[ny][nx] is None:
                    self.propagation_stack.append((nx, ny))
                    
    def _propagate(self):
        """Propagate constraints to maintain consistency"""
        while self.propagation_stack:
            x, y = self.propagation_stack.pop()
            
            if self.collapsed[y][x] is not None:
                continue
                
            # Get current possibilities
            current_possible = self.wave[y][x].copy()
            
            # Check constraints from neighbors
            new_possible = self._get_possible_tiles(x, y)
            
            # Update wave function
            self.wave[y][x] = current_possible.intersection(new_possible)
            
            # If possibilities changed, propagate to neighbors
            if self.wave[y][x] != current_possible:
                self._add_neighbors_to_stack(x, y)
                
    def _get_possible_tiles(self, x: int, y: int) -> Set[int]:
        """Get possible tiles based on neighbor constraints"""
        possible = set(self.tiles.keys())
        
        neighbors = [
            (x, y-1, 'N', 'S'),  # North neighbor constrains via South edge
            (x+1, y, 'E', 'W'),  # East neighbor constrains via West edge
            (x, y+1, 'S', 'N'),  # South neighbor constrains via North edge
            (x-1, y, 'W', 'E')   # West neighbor constrains via East edge
        ]
        
        for nx, ny, our_dir, their_dir in neighbors:
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbor_possible = self.wave[ny][nx]
                
                # Find tiles compatible with neighbor
                compatible = set()
                for neighbor_tile in neighbor_possible:
                    compatible.update(self.adjacency_rules[our_dir][neighbor_tile])
                    
                possible = possible.intersection(compatible)
                
        return possible


class HierarchicalWFC:
    """Hierarchical WFC with multiple levels"""
    
    def __init__(self, config_path: Path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.levels = self.config['levels']
        self.tiles_by_level = {}
        self.neural_constraints = None
        
        # Load tiles for each level
        self._load_tiles()
        
    def _load_tiles(self):
        """Load tile definitions for each level"""
        for level_name, level_config in self.levels.items():
            tiles = []
            
            for tile_def in level_config['tiles']:
                tile = Tile(
                    id=tile_def['id'],
                    name=tile_def['name'],
                    biome=tile_def['biome'],
                    weight=tile_def.get('weight', 1.0),
                    edges=tile_def.get('edges', {})
                )
                tiles.append(tile)
                
            self.tiles_by_level[level_name] = tiles
            
    def generate(self, size: Tuple[int, int], start_level: str = 'region') -> Dict[str, np.ndarray]:
        """Generate hierarchical world"""
        results = {}
        current_size = size
        
        # Generate from top level down
        level_order = ['region', 'biome', 'tile']
        start_idx = level_order.index(start_level)
        
        for level_name in level_order[start_idx:]:
            if level_name not in self.tiles_by_level:
                continue
                
            level_config = self.levels[level_name]
            
            # Create WFC for this level
            wfc = WaveFunctionCollapse(
                tiles=self.tiles_by_level[level_name],
                grid_size=current_size
            )
            
            # Apply constraints from higher level if available
            if level_name != start_level and level_order[level_order.index(level_name) - 1] in results:
                self._apply_hierarchical_constraints(
                    wfc,
                    results[level_order[level_order.index(level_name) - 1]],
                    level_config
                )
                
            # Apply neural constraints if available
            if self.neural_constraints:
                self._apply_neural_constraints(wfc, level_name)
                
            # Collapse
            result = wfc.collapse()
            results[level_name] = result
            
            # Update size for next level
            scale = level_config.get('scale', 2)
            current_size = (current_size[0] * scale, current_size[1] * scale)
            
        return results
        
    def _apply_hierarchical_constraints(self, wfc: WaveFunctionCollapse, 
                                      parent_result: np.ndarray, 
                                      level_config: Dict[str, any]):
        """Apply constraints from parent level"""
        scale = level_config.get('scale', 2)
        
        for y in range(wfc.height):
            for x in range(wfc.width):
                # Find parent cell
                px = x // scale
                py = y // scale
                
                if py < parent_result.shape[0] and px < parent_result.shape[1]:
                    parent_tile = parent_result[py, px]
                    
                    # Filter possibilities based on parent
                    allowed_tiles = set()
                    for tile_id, tile in wfc.tiles.items():
                        if self._is_compatible_with_parent(tile, parent_tile, level_config):
                            allowed_tiles.add(tile_id)
                            
                    wfc.wave[y][x] = wfc.wave[y][x].intersection(allowed_tiles)
                    
    def _is_compatible_with_parent(self, tile: Tile, parent_tile_id: int, 
                                  level_config: Dict[str, any]) -> bool:
        """Check if tile is compatible with parent tile"""
        compatibility = level_config.get('parent_compatibility', {})
        
        if str(parent_tile_id) in compatibility:
            allowed_biomes = compatibility[str(parent_tile_id)]
            return tile.biome in allowed_biomes
            
        return True
        
    def _apply_neural_constraints(self, wfc: WaveFunctionCollapse, level_name: str):
        """Apply learned neural constraints"""
        if self.neural_constraints and hasattr(self.neural_constraints, level_name):
            constraint_model = getattr(self.neural_constraints, level_name)
            
            # Apply constraints (simplified)
            for y in range(wfc.height):
                for x in range(wfc.width):
                    # Get context
                    context = self._get_cell_context(wfc, x, y)
                    
                    # Get probabilities from neural model
                    with torch.no_grad():
                        probs = constraint_model(context)
                        
                    # Update weights based on neural predictions
                    for tile_id in wfc.wave[y][x]:
                        if tile_id < len(probs):
                            wfc.tiles[tile_id].weight *= probs[tile_id].item()
                            
    def _get_cell_context(self, wfc: WaveFunctionCollapse, x: int, y: int) -> torch.Tensor:
        """Get context features for neural constraint model"""
        # Simplified context: one-hot encoding of neighbor possibilities
        context_size = 5  # 5x5 window
        offset = context_size // 2
        
        features = []
        
        for dy in range(-offset, offset + 1):
            for dx in range(-offset, offset + 1):
                nx, ny = x + dx, y + dy
                
                if 0 <= nx < wfc.width and 0 <= ny < wfc.height:
                    # One-hot encode possibilities
                    cell_features = np.zeros(len(wfc.tiles))
                    for tile_id in wfc.wave[ny][nx]:
                        cell_features[tile_id] = 1.0 / len(wfc.wave[ny][nx])
                    features.extend(cell_features)
                else:
                    # Padding
                    features.extend(np.zeros(len(wfc.tiles)))
                    
        return torch.FloatTensor(features).unsqueeze(0)


class BiomeTileRules:
    """Tile pattern rules for different biomes"""
    
    @staticmethod
    def create_forest_tiles() -> List[Tile]:
        """Create forest biome tiles"""
        return [
            Tile(0, "grass", "forest", edges={'N': 'grass', 'E': 'grass', 'S': 'grass', 'W': 'grass'}),
            Tile(1, "tree", "forest", edges={'N': 'tree', 'E': 'tree', 'S': 'tree', 'W': 'tree'}, weight=0.3),
            Tile(2, "path", "forest", edges={'N': 'path', 'E': 'path', 'S': 'path', 'W': 'path'}, weight=0.2),
            Tile(3, "water", "forest", edges={'N': 'water', 'E': 'water', 'S': 'water', 'W': 'water'}, weight=0.1),
            Tile(4, "grass_tree_N", "forest", edges={'N': 'tree', 'E': 'grass', 'S': 'grass', 'W': 'grass'}, weight=0.5),
            Tile(5, "grass_path_E", "forest", edges={'N': 'grass', 'E': 'path', 'S': 'grass', 'W': 'grass'}, weight=0.5),
        ]
        
    @staticmethod
    def create_desert_tiles() -> List[Tile]:
        """Create desert biome tiles"""
        return [
            Tile(10, "sand", "desert", edges={'N': 'sand', 'E': 'sand', 'S': 'sand', 'W': 'sand'}),
            Tile(11, "dune", "desert", edges={'N': 'dune', 'E': 'dune', 'S': 'dune', 'W': 'dune'}, weight=0.3),
            Tile(12, "rock", "desert", edges={'N': 'rock', 'E': 'rock', 'S': 'rock', 'W': 'rock'}, weight=0.2),
            Tile(13, "oasis", "desert", edges={'N': 'water', 'E': 'water', 'S': 'water', 'W': 'water'}, weight=0.05),
        ]
        
    @staticmethod
    def create_snow_tiles() -> List[Tile]:
        """Create snow biome tiles"""
        return [
            Tile(20, "snow", "snow", edges={'N': 'snow', 'E': 'snow', 'S': 'snow', 'W': 'snow'}),
            Tile(21, "ice", "snow", edges={'N': 'ice', 'E': 'ice', 'S': 'ice', 'W': 'ice'}, weight=0.3),
            Tile(22, "pine", "snow", edges={'N': 'pine', 'E': 'pine', 'S': 'pine', 'W': 'pine'}, weight=0.2),
        ]


# Test WFC implementation
if __name__ == "__main__":
    # Create test tiles
    forest_tiles = BiomeTileRules.create_forest_tiles()
    
    # Create WFC instance
    wfc = WaveFunctionCollapse(forest_tiles, grid_size=(32, 32))
    
    # Generate
    try:
        result = wfc.collapse()
        print(f"Generated {result.shape} map")
        
        # Count tile types
        unique, counts = np.unique(result, return_counts=True)
        for tile_id, count in zip(unique, counts):
            tile = wfc.tiles[tile_id]
            print(f"{tile.name}: {count} tiles ({count/result.size*100:.1f}%)")
            
    except ValueError as e:
        print(f"Generation failed: {e}")