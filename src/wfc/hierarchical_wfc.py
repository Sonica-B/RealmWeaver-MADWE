"""
Hierarchical Wave Function Collapse Implementation
Day 3-4: Complete WFC with Unity integration support
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """Represents a tile in the WFC system"""
    id: int
    name: str
    biome: str
    rotation: int = 0
    weight: float = 1.0
    edges: Dict[str, str] = field(default_factory=lambda: {
        'N': 'default', 'E': 'default', 'S': 'default', 'W': 'default'
    })
    unity_prefab: str = ""  # Path to Unity prefab
    
    def rotate(self, times: int = 1) -> 'Tile':
        """Create rotated version of tile"""
        new_edges = self.edges.copy()
        directions = ['N', 'E', 'S', 'W']
        
        for _ in range(times % 4):
            temp = new_edges['N']
            new_edges['N'] = new_edges['W']
            new_edges['W'] = new_edges['S']
            new_edges['S'] = new_edges['E']
            new_edges['E'] = temp
            
        return Tile(
            id=self.id * 10 + times,  # New ID for rotated version
            name=f"{self.name}_rot{times}",
            biome=self.biome,
            rotation=times * 90,
            weight=self.weight,
            edges=new_edges,
            unity_prefab=self.unity_prefab
        )


class WaveFunctionCollapse:
    """Enhanced WFC implementation with Unity support"""
    
    def __init__(self, tiles: List[Tile], grid_size: Tuple[int, int], 
                 enable_rotation: bool = True):
        self.grid_size = grid_size
        self.width, self.height = grid_size
        self.enable_rotation = enable_rotation
        
        # Generate rotated tiles if enabled
        self.tiles = {}
        for tile in tiles:
            self.tiles[tile.id] = tile
            if enable_rotation and tile.name != 'empty':
                for rot in range(1, 4):
                    rotated = tile.rotate(rot)
                    self.tiles[rotated.id] = rotated
        
        # Initialize wave function
        self.reset()
        
        # Compute adjacency rules
        self.adjacency_rules = self._compute_adjacency_rules()
        
    def reset(self):
        """Reset the wave function to initial state"""
        # Each cell contains possible tile IDs
        self.wave = [[set(self.tiles.keys()) for _ in range(self.width)] 
                     for _ in range(self.height)]
        
        # Final tile assignments
        self.collapsed = [[None for _ in range(self.width)] 
                          for _ in range(self.height)]
        
        # Track collapse order for Unity
        self.collapse_order = []
        
    def _compute_adjacency_rules(self) -> Dict[str, Dict[int, Set[int]]]:
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
                        
        return dict(rules)
        
    def get_entropy(self, x: int, y: int) -> float:
        """Calculate entropy (uncertainty) for a cell"""
        possible = self.wave[y][x]
        if len(possible) <= 1:
            return float('inf')
            
        # Shannon entropy with noise to break ties
        weights = [self.tiles[tid].weight for tid in possible]
        total_weight = sum(weights)
        if total_weight == 0:
            return float('inf')
            
        entropy = 0
        for w in weights:
            if w > 0:
                p = w / total_weight
                entropy -= p * np.log(p)
                
        # Add small noise to break ties
        return entropy + random.random() * 0.01
        
    def collapse_cell(self, x: int, y: int) -> bool:
        """Collapse a cell to a single tile"""
        possible = list(self.wave[y][x])
        if not possible:
            return False
            
        # Weight-based selection
        weights = [self.tiles[tid].weight for tid in possible]
        total = sum(weights)
        if total == 0:
            chosen = random.choice(possible)
        else:
            r = random.random() * total
            cumsum = 0
            for tid, w in zip(possible, weights):
                cumsum += w
                if r <= cumsum:
                    chosen = tid
                    break
            else:
                chosen = possible[-1]
        
        # Collapse to chosen tile
        self.wave[y][x] = {chosen}
        self.collapsed[y][x] = chosen
        self.collapse_order.append((x, y, chosen))
        
        return True
        
    def propagate(self, x: int, y: int) -> bool:
        """Propagate constraints from collapsed cell"""
        stack = [(x, y)]
        
        while stack:
            cx, cy = stack.pop()
            current_possible = self.wave[cy][cx]
            
            # Check each neighbor
            neighbors = [
                (cx, cy-1, 'S', 'N'),  # North
                (cx+1, cy, 'W', 'E'),  # East  
                (cx, cy+1, 'N', 'S'),  # South
                (cx-1, cy, 'E', 'W')   # West
            ]
            
            for nx, ny, our_dir, their_dir in neighbors:
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbor_possible = self.wave[ny][nx].copy()
                    
                    # Find valid tiles for neighbor
                    valid = set()
                    for our_tile in current_possible:
                        valid.update(self.adjacency_rules[their_dir].get(our_tile, set()))
                    
                    # Constrain neighbor
                    new_possible = neighbor_possible & valid
                    
                    if len(new_possible) == 0:
                        return False  # Contradiction
                        
                    if len(new_possible) < len(neighbor_possible):
                        self.wave[ny][nx] = new_possible
                        stack.append((nx, ny))
                        
        return True
        
    def find_min_entropy_cell(self) -> Optional[Tuple[int, int]]:
        """Find uncollapsed cell with minimum entropy"""
        min_entropy = float('inf')
        min_cell = None
        
        for y in range(self.height):
            for x in range(self.width):
                if self.collapsed[y][x] is None:
                    entropy = self.get_entropy(x, y)
                    if entropy < min_entropy:
                        min_entropy = entropy
                        min_cell = (x, y)
                        
        return min_cell
        
    def collapse(self) -> np.ndarray:
        """Run the WFC algorithm"""
        iterations = 0
        max_iterations = self.width * self.height * 10
        
        while True:
            # Find cell with minimum entropy
            cell = self.find_min_entropy_cell()
            if cell is None:
                break  # All cells collapsed
                
            x, y = cell
            
            # Collapse the cell
            if not self.collapse_cell(x, y):
                logger.warning(f"Failed to collapse cell ({x}, {y})")
                return None
                
            # Propagate constraints
            if not self.propagate(x, y):
                logger.warning(f"Contradiction during propagation from ({x}, {y})")
                return None
                
            iterations += 1
            if iterations > max_iterations:
                logger.error("Max iterations reached")
                return None
                
        # Convert to numpy array
        result = np.array([[self.collapsed[y][x] for x in range(self.width)] 
                          for y in range(self.height)])
        
        return result
    
    def get_unity_data(self) -> Dict[str, Any]:
        """Get data formatted for Unity"""
        tiles_data = []
        for y in range(self.height):
            for x in range(self.width):
                tile_id = self.collapsed[y][x]
                if tile_id is not None:
                    tile = self.tiles[tile_id]
                    tiles_data.append({
                        'x': x,
                        'y': y,
                        'id': tile_id,
                        'name': tile.name,
                        'biome': tile.biome,
                        'rotation': tile.rotation,
                        'prefab': tile.unity_prefab
                    })
                    
        return {
            'width': self.width,
            'height': self.height,
            'tiles': tiles_data,
            'collapse_order': [(x, y, tid) for x, y, tid in self.collapse_order]
        }


class BiomeTileRules:
    """Predefined tile rules for different biomes"""
    
    @staticmethod
    def get_forest_tiles() -> List[Tile]:
        """Get forest biome tiles"""
        return [
            Tile(1, "grass", "forest", weight=3.0, 
                 edges={'N': 'grass', 'E': 'grass', 'S': 'grass', 'W': 'grass'},
                 unity_prefab="Tiles/Forest/Grass"),
            Tile(2, "tree", "forest", weight=1.0,
                 edges={'N': 'grass', 'E': 'grass', 'S': 'grass', 'W': 'grass'},
                 unity_prefab="Tiles/Forest/Tree"),
            Tile(3, "path", "forest", weight=0.5,
                 edges={'N': 'path', 'E': 'grass', 'S': 'path', 'W': 'grass'},
                 unity_prefab="Tiles/Forest/Path"),
            Tile(4, "water", "forest", weight=0.3,
                 edges={'N': 'water', 'E': 'water', 'S': 'water', 'W': 'water'},
                 unity_prefab="Tiles/Forest/Water"),
        ]
    
    @staticmethod
    def get_desert_tiles() -> List[Tile]:
        """Get desert biome tiles"""
        return [
            Tile(11, "sand", "desert", weight=3.0,
                 edges={'N': 'sand', 'E': 'sand', 'S': 'sand', 'W': 'sand'},
                 unity_prefab="Tiles/Desert/Sand"),
            Tile(12, "dune", "desert", weight=1.0,
                 edges={'N': 'sand', 'E': 'sand', 'S': 'sand', 'W': 'sand'},
                 unity_prefab="Tiles/Desert/Dune"),
            Tile(13, "cactus", "desert", weight=0.3,
                 edges={'N': 'sand', 'E': 'sand', 'S': 'sand', 'W': 'sand'},
                 unity_prefab="Tiles/Desert/Cactus"),
            Tile(14, "oasis", "desert", weight=0.1,
                 edges={'N': 'water', 'E': 'water', 'S': 'water', 'W': 'water'},
                 unity_prefab="Tiles/Desert/Oasis"),
        ]
    
    @staticmethod
    def get_cyberpunk_tiles() -> List[Tile]:
        """Get cyberpunk biome tiles"""
        return [
            Tile(21, "street", "cyberpunk", weight=2.0,
                 edges={'N': 'road', 'E': 'road', 'S': 'road', 'W': 'road'},
                 unity_prefab="Tiles/Cyberpunk/Street"),
            Tile(22, "building", "cyberpunk", weight=1.5,
                 edges={'N': 'wall', 'E': 'wall', 'S': 'road', 'W': 'wall'},
                 unity_prefab="Tiles/Cyberpunk/Building"),
            Tile(23, "neon", "cyberpunk", weight=0.5,
                 edges={'N': 'wall', 'E': 'road', 'S': 'road', 'W': 'road'},
                 unity_prefab="Tiles/Cyberpunk/Neon"),
            Tile(24, "plaza", "cyberpunk", weight=0.3,
                 edges={'N': 'road', 'E': 'road', 'S': 'road', 'W': 'road'},
                 unity_prefab="Tiles/Cyberpunk/Plaza"),
        ]

# Add this to src/generation/wfc/hierarchical_wfc.py

class BiomeTileRules:
    """Tile rules for different biomes"""
    
    @staticmethod
    def get_forest_tiles() -> List[Tile]:
        return [
            Tile(0, "grass", "forest", weight=3.0, 
                 edges={'N': 'grass', 'E': 'grass', 'S': 'grass', 'W': 'grass'}),
            Tile(1, "tree", "forest", weight=1.0,
                 edges={'N': 'tree', 'E': 'tree', 'S': 'tree', 'W': 'tree'}),
            Tile(2, "path", "forest", weight=0.5,
                 edges={'N': 'path', 'E': 'grass', 'S': 'path', 'W': 'grass'}),
            Tile(3, "water", "forest", weight=0.2,
                 edges={'N': 'water', 'E': 'water', 'S': 'water', 'W': 'water'}),
            Tile(4, "grass_tree", "forest", weight=0.8,
                 edges={'N': 'tree', 'E': 'grass', 'S': 'grass', 'W': 'grass'}),
        ]
    
    @staticmethod
    def get_desert_tiles() -> List[Tile]:
        return [
            Tile(10, "sand", "desert", weight=3.0,
                 edges={'N': 'sand', 'E': 'sand', 'S': 'sand', 'W': 'sand'}),
            Tile(11, "dune", "desert", weight=1.0,
                 edges={'N': 'dune', 'E': 'sand', 'S': 'dune', 'W': 'sand'}),
            Tile(12, "cactus", "desert", weight=0.3,
                 edges={'N': 'sand', 'E': 'sand', 'S': 'sand', 'W': 'sand'}),
            Tile(13, "oasis", "desert", weight=0.1,
                 edges={'N': 'water', 'E': 'water', 'S': 'water', 'W': 'water'}),
        ]
    
    @staticmethod
    def get_cyberpunk_tiles() -> List[Tile]:
        return [
            Tile(20, "street", "cyberpunk", weight=2.0,
                 edges={'N': 'road', 'E': 'road', 'S': 'road', 'W': 'road'}),
            Tile(21, "building", "cyberpunk", weight=1.5,
                 edges={'N': 'wall', 'E': 'wall', 'S': 'road', 'W': 'wall'}),
            Tile(22, "neon", "cyberpunk", weight=0.5,
                 edges={'N': 'wall', 'E': 'road', 'S': 'road', 'W': 'road'}),
        ]


class HierarchicalWFC:
    """Hierarchical WFC with multiple generation levels"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.levels = {
            'region': {'scale': 4, 'size': (8, 8)},
            'biome': {'scale': 4, 'size': (32, 32)},
            'tile': {'scale': 1, 'size': (32, 32)}
        }
        
        # Biome tile configurations
        self.biome_tiles = {
            'forest': BiomeTileRules.get_forest_tiles(),
            'desert': BiomeTileRules.get_desert_tiles(),
            'cyberpunk': BiomeTileRules.get_cyberpunk_tiles()
        }
        
        if config_path and config_path.exists():
            self._load_config(config_path)
            
    def _load_config(self, config_path: Path):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.levels.update(config.get('levels', {}))
            
    def generate_region(self, size: Tuple[int, int]) -> np.ndarray:
        """Generate high-level region map"""
        # Simple region generation for now
        region_tiles = [
            Tile(101, "forest_region", "region", weight=1.0),
            Tile(102, "desert_region", "region", weight=0.8),
            Tile(103, "cyberpunk_region", "region", weight=0.6),
        ]
        
        wfc = WaveFunctionCollapse(region_tiles, size, enable_rotation=False)
        return wfc.collapse()
        
    def generate_biome(self, region_map: np.ndarray, 
                      position: Tuple[int, int]) -> Tuple[str, np.ndarray]:
        """Generate biome-level map based on region"""
        # Map region to biome
        region_to_biome = {
            101: 'forest',
            102: 'desert', 
            103: 'cyberpunk'
        }
        
        rx, ry = position[0] // 4, position[1] // 4
        if 0 <= rx < region_map.shape[1] and 0 <= ry < region_map.shape[0]:
            region_id = region_map[ry, rx]
            biome = region_to_biome.get(region_id, 'forest')
        else:
            biome = 'forest'
            
        # Generate biome tiles
        tiles = self.biome_tiles.get(biome, self.biome_tiles['forest'])
        wfc = WaveFunctionCollapse(tiles, self.levels['tile']['size'])
        
        result = wfc.collapse()
        if result is None:
            # Fallback to simple generation
            result = np.full(self.levels['tile']['size'], tiles[0].id)
            
        return biome, result
        
    def generate_full(self, world_size: Tuple[int, int]) -> Dict[str, Any]:
        """Generate complete hierarchical world"""
        # Generate region map
        region_size = (world_size[0] // 32, world_size[1] // 32)
        region_map = self.generate_region(region_size)
        
        # Generate biome maps
        biome_maps = {}
        for y in range(0, world_size[1], 32):
            for x in range(0, world_size[0], 32):
                chunk_pos = (x, y)
                biome, tile_map = self.generate_biome(region_map, chunk_pos)
                biome_maps[chunk_pos] = {
                    'biome': biome,
                    'tiles': tile_map,
                    'position': chunk_pos
                }
                
        return {
            'region_map': region_map.tolist(),
            'biome_maps': biome_maps,
            'world_size': world_size
        }


# Test function
def test_wfc():
    """Test WFC generation"""
    print("Testing WFC generation...")
    
    # Test basic WFC
    forest_tiles = BiomeTileRules.get_forest_tiles()
    wfc = WaveFunctionCollapse(forest_tiles, (16, 16))
    result = wfc.collapse()
    
    if result is not None:
        print(f"Generated {wfc.width}x{wfc.height} map successfully")
        print(f"Unique tiles: {np.unique(result)}")
        
        # Get Unity data
        unity_data = wfc.get_unity_data()
        print(f"Unity data contains {len(unity_data['tiles'])} tiles")
    else:
        print("Generation failed!")
        
    # Test hierarchical generation
    print("\nTesting hierarchical generation...")
    hwfc = HierarchicalWFC()
    world = hwfc.generate_full((64, 64))
    print(f"Generated world with {len(world['biome_maps'])} chunks")


if __name__ == "__main__":
    test_wfc()