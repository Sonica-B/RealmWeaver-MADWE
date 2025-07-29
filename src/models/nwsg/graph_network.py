"""
Neural World State Graph (NWSG) Implementation
Day 7 - Production Code
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import time
import pickle
import threading
from enum import Enum
import torch
import torch.nn as nn


class NodeType(Enum):
    REGION = "region"
    BIOME = "biome"
    CHUNK = "chunk"
    ENTITY = "entity"
    AGENT = "agent"
    PLAYER = "player"


@dataclass
class WorldNode:
    """Represents a node in the world state graph"""
    node_id: str
    node_type: NodeType
    position: Tuple[float, float, float]
    bounds: Tuple[float, float, float, float]
    attributes: Dict[str, Any] = field(default_factory=dict)
    connections: Set[str] = field(default_factory=set)
    timestamp: float = field(default_factory=time.time)
    version: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "position": self.position,
            "bounds": self.bounds,
            "attributes": self.attributes,
            "connections": list(self.connections),
            "timestamp": self.timestamp,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorldNode':
        node = cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            position=tuple(data["position"]),
            bounds=tuple(data["bounds"]),
            attributes=data["attributes"],
            connections=set(data["connections"]),
            timestamp=data["timestamp"],
            version=data["version"]
        )
        return node


@dataclass
class StateSnapshot:
    """Represents a snapshot of the world state"""
    snapshot_id: str
    timestamp: float
    nodes: Dict[str, WorldNode]
    spatial_index_data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpatialIndex:
    """Simple spatial indexing for efficient queries"""
    
    def __init__(self):
        self.node_map = {}
        self.grid_size = 100
        self.grid = defaultdict(list)
        self._next_id = 0
        
    def insert(self, node: WorldNode):
        """Insert node into spatial index"""
        idx_id = self._next_id
        self._next_id += 1
        
        self.node_map[idx_id] = node.node_id
        
        # Add to grid cells
        cells = self._get_cells(node.bounds)
        for cell in cells:
            self.grid[cell].append(idx_id)
        
    def delete(self, node: WorldNode):
        """Remove node from spatial index"""
        idx_id = None
        for iid, nid in self.node_map.items():
            if nid == node.node_id:
                idx_id = iid
                break
        
        if idx_id is not None:
            # Remove from grid
            cells = self._get_cells(node.bounds)
            for cell in cells:
                if idx_id in self.grid[cell]:
                    self.grid[cell].remove(idx_id)
            del self.node_map[idx_id]
    
    def query_region(self, bounds: Tuple[float, float, float, float]) -> List[str]:
        """Query nodes within a bounding box"""
        results = set()
        cells = self._get_cells(bounds)
        
        for cell in cells:
            for idx_id in self.grid[cell]:
                if idx_id in self.node_map:
                    results.add(self.node_map[idx_id])
        
        return list(results)
    
    def nearest_neighbors(self, point: Tuple[float, float], k: int = 5) -> List[str]:
        """Find k nearest neighbors to a point"""
        # Simplified - check nearby cells
        cell = self._point_to_cell(point)
        candidates = []
        
        # Check surrounding cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nearby_cell = (cell[0] + dx, cell[1] + dy)
                for idx_id in self.grid.get(nearby_cell, []):
                    if idx_id in self.node_map:
                        candidates.append(self.node_map[idx_id])
        
        return candidates[:k]
    
    def _get_cells(self, bounds: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """Get grid cells that bounds overlaps"""
        min_x, min_y, max_x, max_y = bounds
        cells = []
        
        start_x = int(min_x // self.grid_size)
        start_y = int(min_y // self.grid_size)
        end_x = int(max_x // self.grid_size) + 1
        end_y = int(max_y // self.grid_size) + 1
        
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                cells.append((x, y))
        
        return cells
    
    def _point_to_cell(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert point to grid cell"""
        return (int(point[0] // self.grid_size), int(point[1] // self.grid_size))
    
    def get_state(self) -> bytes:
        """Serialize spatial index state"""
        return pickle.dumps({
            'node_map': self.node_map,
            'next_id': self._next_id,
            'grid': dict(self.grid)
        })
    
    def load_state(self, state_data: bytes, nodes: Dict[str, WorldNode]):
        """Load spatial index from serialized state"""
        state = pickle.loads(state_data)
        self.node_map = state['node_map']
        self._next_id = state['next_id']
        self.grid = defaultdict(list, state['grid'])


class NeuralWorldStateGraph:
    """Neural World State Graph with spatial indexing and state management"""
    
    def __init__(self, embedding_dim: int = 128):
        # Core graph structure
        self.nodes: Dict[str, WorldNode] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        
        # Spatial indexing
        self.spatial_index = SpatialIndex()
        
        # State management
        self.snapshots: Dict[str, StateSnapshot] = {}
        self.current_version = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Neural components
        self.embedding_dim = embedding_dim
        self.node_embeddings = {}
        self.transformer = self._init_transformer()
        
        # Performance monitoring
        self.query_times = deque(maxlen=1000)
        self.update_times = deque(maxlen=1000)
        
    def _init_transformer(self) -> nn.Module:
        """Initialize transformer for cross-agent communication"""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1
            ),
            num_layers=3
        )
    
    def add_node(self, node: WorldNode) -> bool:
        """Add a node to the graph"""
        with self.lock:
            start_time = time.time()
            
            if node.node_id in self.nodes:
                return False
            
            # Add to graph
            self.nodes[node.node_id] = node
            
            # Add to spatial index
            self.spatial_index.insert(node)
            
            # Initialize embeddings
            self.node_embeddings[node.node_id] = torch.randn(self.embedding_dim)
            
            # Update version
            node.version = self.current_version
            self.current_version += 1
            
            # Track performance
            self.update_times.append(time.time() - start_time)
            
            return True
    
    def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update node attributes"""
        with self.lock:
            start_time = time.time()
            
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            
            # Update attributes
            node.attributes.update(updates)
            node.timestamp = time.time()
            node.version = self.current_version
            self.current_version += 1
            
            # Update spatial index if position changed
            if 'position' in updates or 'bounds' in updates:
                self.spatial_index.delete(node)
                if 'position' in updates:
                    node.position = updates['position']
                if 'bounds' in updates:
                    node.bounds = updates['bounds']
                self.spatial_index.insert(node)
            
            self.update_times.append(time.time() - start_time)
            
            return True
    
    def add_edge(self, from_id: str, to_id: str, bidirectional: bool = True):
        """Add edge between nodes"""
        with self.lock:
            if from_id in self.nodes and to_id in self.nodes:
                self.edges[from_id].add(to_id)
                self.nodes[from_id].connections.add(to_id)
                
                if bidirectional:
                    self.edges[to_id].add(from_id)
                    self.nodes[to_id].connections.add(from_id)
    
    def query_spatial(self, bounds: Tuple[float, float, float, float], 
                     node_types: Optional[List[NodeType]] = None) -> List[WorldNode]:
        """Query nodes within spatial bounds"""
        with self.lock:
            start_time = time.time()
            
            # Get nodes from spatial index
            node_ids = self.spatial_index.query_region(bounds)
            
            # Filter by type if specified
            results = []
            for node_id in node_ids:
                node = self.nodes.get(node_id)
                if node and (node_types is None or node.node_type in node_types):
                    results.append(node)
            
            self.query_times.append(time.time() - start_time)
            
            return results
    
    def find_nearest(self, position: Tuple[float, float], 
                    k: int = 5, node_types: Optional[List[NodeType]] = None) -> List[WorldNode]:
        """Find k nearest nodes to a position"""
        with self.lock:
            start_time = time.time()
            
            # Get candidates
            candidates = self.spatial_index.nearest_neighbors(position, k * 3)
            
            # Filter and sort by actual distance
            results = []
            for node_id in candidates:
                node = self.nodes.get(node_id)
                if node and (node_types is None or node.node_type in node_types):
                    dist = np.sqrt((node.position[0] - position[0])**2 + 
                                 (node.position[1] - position[1])**2)
                    results.append((dist, node))
            
            # Sort by distance and return top k
            results.sort(key=lambda x: x[0])
            
            self.query_times.append(time.time() - start_time)
            
            return [node for _, node in results[:k]]
    
    def get_connected_nodes(self, node_id: str, max_depth: int = 1) -> Set[str]:
        """Get all nodes connected within max_depth steps"""
        with self.lock:
            if node_id not in self.nodes:
                return set()
            
            visited = set()
            queue = deque([(node_id, 0)])
            
            while queue:
                current_id, depth = queue.popleft()
                
                if current_id in visited or depth > max_depth:
                    continue
                
                visited.add(current_id)
                
                if depth < max_depth:
                    for neighbor_id in self.edges.get(current_id, set()):
                        if neighbor_id not in visited:
                            queue.append((neighbor_id, depth + 1))
            
            return visited
    
    def create_snapshot(self, snapshot_id: str) -> StateSnapshot:
        """Create a snapshot of current state"""
        with self.lock:
            snapshot = StateSnapshot(
                snapshot_id=snapshot_id,
                timestamp=time.time(),
                nodes={nid: WorldNode.from_dict(node.to_dict()) 
                       for nid, node in self.nodes.items()},
                spatial_index_data=self.spatial_index.get_state(),
                metadata={
                    "version": self.current_version,
                    "node_count": len(self.nodes),
                    "edge_count": sum(len(edges) for edges in self.edges.values())
                }
            )
            
            self.snapshots[snapshot_id] = snapshot
            return snapshot
    
    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rollback to a previous snapshot"""
        with self.lock:
            if snapshot_id not in self.snapshots:
                return False
            
            snapshot = self.snapshots[snapshot_id]
            
            # Clear current state
            self.nodes.clear()
            self.edges.clear()
            self.node_embeddings.clear()
            
            # Restore nodes
            for node_id, node_data in snapshot.nodes.items():
                self.nodes[node_id] = node_data
                self.node_embeddings[node_id] = torch.randn(self.embedding_dim)
                
                # Restore edges
                for connected_id in node_data.connections:
                    self.edges[node_id].add(connected_id)
            
            # Restore spatial index
            self.spatial_index = SpatialIndex()
            self.spatial_index.load_state(snapshot.spatial_index_data, self.nodes)
            
            # Update version
            self.current_version = snapshot.metadata["version"]
            
            return True
    
    def compute_node_embeddings(self, node_ids: List[str]) -> torch.Tensor:
        """Compute embeddings for nodes using transformer"""
        with self.lock:
            # Gather embeddings
            embeddings = []
            valid_ids = []
            
            for node_id in node_ids:
                if node_id in self.node_embeddings:
                    embeddings.append(self.node_embeddings[node_id])
                    valid_ids.append(node_id)
            
            if not embeddings:
                return torch.zeros(0, self.embedding_dim)
            
            # Stack and process through transformer
            x = torch.stack(embeddings).unsqueeze(1)
            
            with torch.no_grad():
                output = self.transformer(x)
            
            return output.squeeze(1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance and state statistics"""
        with self.lock:
            avg_query_time = np.mean(self.query_times) if self.query_times else 0
            avg_update_time = np.mean(self.update_times) if self.update_times else 0
            
            node_type_counts = defaultdict(int)
            for node in self.nodes.values():
                node_type_counts[node.node_type.value] += 1
            
            return {
                "total_nodes": len(self.nodes),
                "total_edges": sum(len(edges) for edges in self.edges.values()),
                "node_types": dict(node_type_counts),
                "snapshots": len(self.snapshots),
                "current_version": self.current_version,
                "avg_query_time_ms": avg_query_time * 1000,
                "avg_update_time_ms": avg_update_time * 1000,
                "memory_usage_mb": len(pickle.dumps(self.nodes)) / (1024 * 1024)
            }
    
    def save_to_file(self, filepath: str):
        """Save graph state to file"""
        with self.lock:
            state = {
                "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
                "embeddings": {nid: emb.tolist() for nid, emb in self.node_embeddings.items()},
                "version": self.current_version,
                "statistics": self.get_statistics()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)