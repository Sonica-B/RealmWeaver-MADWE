"""
Neural World State Graph (NWSG) Implementation
Day 5: Basic implementation for multi-agent coordination
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict
import json


@dataclass
class WorldNode:
    """Node in the world state graph"""
    node_id: str
    node_type: str  # terrain_chunk, asset, character, event
    position: Tuple[float, float, float]
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    connections: Set[str] = field(default_factory=set)


class NeuralWorldStateGraph:
    """Neural World State Graph for maintaining global game state"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.d_model = config.get('d_model', 512)
        self.d_agent = config.get('d_agent', 256)
        self.cell_size = config.get('cell_size', 10.0)
        self.max_temporal_length = config.get('max_temporal_length', 100)
        
        # Graph storage
        self.nodes: Dict[str, WorldNode] = {}
        self.spatial_index: Dict[Tuple[int, int, int], Set[str]] = defaultdict(set)
        self.temporal_buffer: List[Dict[str, Any]] = []
        
        # State tracking
        self.global_state = {
            'total_nodes': 0,
            'nodes_by_type': defaultdict(int),
            'last_update': datetime.now().isoformat(),
            'active_regions': set()
        }
        
    async def add_node(self, node: WorldNode):
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        
        # Update spatial index
        grid_pos = self._get_grid_position(node.position)
        self.spatial_index[grid_pos].add(node.node_id)
        
        # Update state
        self.global_state['total_nodes'] += 1
        self.global_state['nodes_by_type'][node.node_type] += 1
        self.global_state['last_update'] = datetime.now().isoformat()
        
        # Add to temporal buffer
        self._add_to_temporal_buffer({
            'action': 'add_node',
            'node_id': node.node_id,
            'timestamp': node.timestamp
        })
        
    async def remove_node(self, node_id: str):
        """Remove a node from the graph"""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        # Remove from spatial index
        grid_pos = self._get_grid_position(node.position)
        self.spatial_index[grid_pos].discard(node_id)
        
        # Update state
        self.global_state['total_nodes'] -= 1
        self.global_state['nodes_by_type'][node.node_type] -= 1
        
        del self.nodes[node_id]
        
    async def query_spatial(self, position: Tuple[float, float, float], 
                          radius: float) -> List[WorldNode]:
        """Query nodes within radius of position"""
        results = []
        center_grid = self._get_grid_position(position)
        grid_radius = int(np.ceil(radius / self.cell_size))
        
        # Check nearby grid cells
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                for dz in range(-grid_radius, grid_radius + 1):
                    grid_pos = (
                        center_grid[0] + dx,
                        center_grid[1] + dy,
                        center_grid[2] + dz
                    )
                    
                    for node_id in self.spatial_index.get(grid_pos, set()):
                        node = self.nodes[node_id]
                        # Check actual distance
                        dist = np.linalg.norm(
                            np.array(node.position) - np.array(position)
                        )
                        if dist <= radius:
                            results.append(node)
                            
        return results
        
    async def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current world state"""
        active_regions = set()
        
        # Determine active regions from nodes
        for node in self.nodes.values():
            if node.node_type == "terrain_chunk":
                region_x = int(node.position[0] // 64)
                region_z = int(node.position[2] // 64)
                active_regions.add((region_x, region_z))
                
        summary = {
            'total_nodes': self.global_state['total_nodes'],
            'nodes_by_type': dict(self.global_state['nodes_by_type']),
            'active_regions': list(active_regions),
            'last_update': self.global_state['last_update'],
            'temporal_buffer_size': len(self.temporal_buffer)
        }
        
        return summary
        
    def _get_grid_position(self, position: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert world position to grid coordinates"""
        return (
            int(position[0] // self.cell_size),
            int(position[1] // self.cell_size),
            int(position[2] // self.cell_size)
        )
        
    def _add_to_temporal_buffer(self, event: Dict[str, Any]):
        """Add event to temporal buffer"""
        self.temporal_buffer.append(event)
        
        # Maintain buffer size
        if len(self.temporal_buffer) > self.max_temporal_length:
            self.temporal_buffer.pop(0)
            
    async def get_temporal_state(self, time_window: float) -> List[Dict[str, Any]]:
        """Get events within time window"""
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - time_window
        
        return [
            event for event in self.temporal_buffer
            if event.get('timestamp', 0) >= cutoff_time
        ]