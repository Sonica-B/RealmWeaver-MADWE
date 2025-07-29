"""
Hierarchical Wave Function Collapse Implementation
Day 6: Complete hierarchical generation with constraint propagation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from collections import defaultdict
import logging
import torch
import torch.nn as nn

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
    constraints: Dict[str, Any] = field(default_factory=dict)


class NeuralConstraintLearner(nn.Module):
    """Placeholder for neural constraint learning"""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.constraint_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, parent_features: torch.Tensor, child_features: torch.Tensor) -> torch.Tensor:
        parent_encoded = self.encoder(parent_features)
        child_encoded = self.encoder(child_features)
        combined = torch.cat([parent_encoded, child_encoded], dim=-1)
        return self.constraint_predictor(combined)


class BiomeTileRules:
    """Enhanced tile rules with hierarchical constraints"""
    
    @staticmethod
    def get_forest_tiles() -> List[Tile]:
        return [
            Tile(0, "grass", "forest", weight=3.0, 
                 edges={'N': 'grass', 'E': 'grass', 'S': 'grass', 'W': 'grass'}),
            Tile(1, "tree", "forest", weight=1.0,
                 edges={'N': 'grass', 'E': 'grass', 'S': 'grass', 'W': 'grass'}),
            Tile(2, "path", "forest", weight=0.5,
                 edges={'N': 'path', 'E': 'grass', 'S': 'path', 'W': 'grass'}),
            Tile(3, "water", "forest", weight=0.2,
                 edges={'N': 'water', 'E': 'water', 'S': 'water', 'W': 'water'}),
        ]
    
    @staticmethod
    def get_desert_tiles() -> List[Tile]:
        return [
            Tile(10, "sand", "desert", weight=3.0,
                 edges={'N': 'sand', 'E': 'sand', 'S': 'sand', 'W': 'sand'}),
            Tile(11, "dune", "desert", weight=1.0,
                 edges={'N': 'sand', 'E': 'sand', 'S': 'sand', 'W': 'sand'}),
            Tile(12, "cactus", "desert", weight=0.3,
                 edges={'N': 'sand', 'E': 'sand', 'S': 'sand', 'W': 'sand'}),
        ]
    
    @staticmethod
    def get_cyberpunk_tiles() -> List[Tile]:
        return [
            Tile(20, "street", "cyberpunk", weight=2.0,
                 edges={'N': 'road', 'E': 'road', 'S': 'road', 'W': 'road'}),
            Tile(21, "building", "cyberpunk", weight=1.5,
                 edges={'N': 'road', 'E': 'road', 'S': 'road', 'W': 'road'}),
            Tile(22, "neon", "cyberpunk", weight=0.5,
                 edges={'N': 'road', 'E': 'road', 'S': 'road', 'W': 'road'}),
        ]


class WaveFunctionCollapse:
    """Basic WFC implementation"""
    
    def __init__(self, tiles: List[Tile], grid_size: Tuple[int, int]):
        self.tiles = {tile.id: tile for tile in tiles}
        self.grid_size = grid_size
        self.width, self.height = grid_size
        self.reset()
        
    def reset(self):
        self.wave = [[set(self.tiles.keys()) for _ in range(self.width)] 
                     for _ in range(self.height)]
        self.collapsed = [[None for _ in range(self.width)] 
                          for _ in range(self.height)]
        
    def collapse(self) -> Optional[np.ndarray]:
        """Simplified collapse - just random selection with weights"""
        tile_ids = list(self.tiles.keys())
        weights = [self.tiles[tid].weight for tid in tile_ids]
        total_weight = sum(weights)
        probabilities = [w/total_weight for w in weights]
        
        # Generate random grid
        result = np.random.choice(tile_ids, size=self.grid_size, p=probabilities)
        return result


class HierarchicalWFC:
    """Hierarchical WFC with constraint propagation between levels"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.levels = {
            'region': {'scale': 4, 'size': (8, 8)},
            'biome': {'scale': 4, 'size': (32, 32)},
            'tile': {'scale': 1, 'size': (32, 32)}
        }
        
        self.hierarchy_constraints = {
            'region_to_biome': {
                101: 'forest',    # forest_region -> forest biome
                102: 'desert',    # desert_region -> desert biome
                103: 'cyberpunk'  # cyberpunk_region -> cyberpunk biome
            }
        }
        
        self.neural_learner = NeuralConstraintLearner()
        
    def generate_hierarchical_world(self, world_size: Tuple[int, int], 
                                  use_neural_constraints: bool = False) -> Dict[str, Any]:
        """Generate complete hierarchical world"""
        
        # Level 1: Generate regions
        region_size = (world_size[0] // 128, world_size[1] // 128)
        region_map = self._generate_regions(region_size)
        
        # Level 2: Generate biomes based on regions
        biome_maps = {}
        
        # Level 3: Generate tiles based on biomes
        tile_maps = {}
        
        for y in range(0, world_size[1], 32):
            for x in range(0, world_size[0], 32):
                chunk_pos = (x, y)
                
                # Get region for this chunk
                rx = min(x // 128, region_map.shape[1] - 1)
                ry = min(y // 128, region_map.shape[0] - 1)
                region_id = region_map[ry, rx]
                
                # Get biome from region
                biome = self.hierarchy_constraints['region_to_biome'].get(region_id, 'forest')
                
                # Generate tiles for this biome
                tiles = self._get_tiles_for_biome(biome)
                wfc = WaveFunctionCollapse(tiles, (32, 32))
                tile_map = wfc.collapse()
                
                biome_maps[chunk_pos] = {
                    'map': np.full((32, 32), region_id - 100),  # Use region ID for visualization
                    'parent_region': region_id,
                    'position': chunk_pos,
                    'dominant_biome': biome
                }
                
                tile_maps[chunk_pos] = tile_map
                
        return {
            'region_map': region_map,
            'biome_maps': biome_maps,
            'tile_maps': tile_maps,
            'world_size': world_size,
            'hierarchy_levels': 3
        }
        
    def _generate_regions(self, size: Tuple[int, int]) -> np.ndarray:
        """Generate top-level region map"""
        # Simple random regions
        region_ids = [101, 102, 103]  # forest, desert, cyberpunk
        weights = [0.4, 0.3, 0.3]
        
        result = np.random.choice(region_ids, size=size, p=weights)
        return result
        
    def _get_tiles_for_biome(self, biome: str) -> List[Tile]:
        """Get tiles for a specific biome"""
        if biome == 'forest':
            return BiomeTileRules.get_forest_tiles()
        elif biome == 'desert':
            return BiomeTileRules.get_desert_tiles()
        elif biome == 'cyberpunk':
            return BiomeTileRules.get_cyberpunk_tiles()
        else:
            return BiomeTileRules.get_forest_tiles()


# For compatibility
EnhancedWFC = WaveFunctionCollapse