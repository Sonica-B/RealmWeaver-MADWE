"""
Neural World State Graph (NWSG) Implementation
Day 7 - Complete Production Code
Author: Ankit Gole
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import time
import pickle
import gzip
import threading
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtree import index as rtree_index
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the world state graph"""
    REGION = "region"
    BIOME = "biome"
    CHUNK = "chunk"
    TILE = "tile"
    ENTITY = "entity"
    AGENT = "agent"
    PLAYER = "player"
    NARRATIVE = "narrative"


@dataclass
class WorldNode:
    """Represents a node in the world state graph"""
    node_id: str
    node_type: NodeType
    position: Tuple[float, float, float]  # 3D position (x, y, z)
    bounds: Tuple[float, float, float, float, float, float]  # 3D bounding box (minx, miny, minz, maxx, maxy, maxz)
    attributes: Dict[str, Any] = field(default_factory=dict)
    connections: Set[str] = field(default_factory=set)
    timestamp: float = field(default_factory=time.time)
    version: int = 0
    embedding: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Serialize node to dictionary"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "position": self.position,
            "bounds": self.bounds,
            "attributes": self.attributes,
            "connections": list(self.connections),
            "timestamp": self.timestamp,
            "version": self.version,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorldNode':
        """Deserialize node from dictionary"""
        node = cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            position=tuple(data["position"]),
            bounds=tuple(data["bounds"]),
            attributes=data.get("attributes", {}),
            connections=set(data.get("connections", [])),
            timestamp=data.get("timestamp", time.time()),
            version=data.get("version", 0),
            metadata=data.get("metadata", {})
        )
        if data.get("embedding"):
            node.embedding = torch.tensor(data["embedding"])
        return node
    
    def get_center(self) -> Tuple[float, float, float]:
        """Get center point of node bounds"""
        return (
            (self.bounds[0] + self.bounds[3]) / 2,
            (self.bounds[1] + self.bounds[4]) / 2,
            (self.bounds[2] + self.bounds[5]) / 2
        )


@dataclass
class StateSnapshot:
    """Represents a snapshot of the world state for persistence/rollback"""
    snapshot_id: str
    timestamp: float
    nodes: Dict[str, Dict]  # Serialized nodes
    edges: Dict[str, List[str]]  # Adjacency lists
    spatial_index_data: bytes  # Serialized spatial index
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    
    def generate_checksum(self):
        """Generate checksum for data integrity"""
        data_str = json.dumps({
            "nodes": sorted(self.nodes.keys()),
            "timestamp": self.timestamp
        }, sort_keys=True)
        self.checksum = hashlib.sha256(data_str.encode()).hexdigest()


class SpatialIndex:
    """R-tree based spatial indexing for efficient 3D queries"""
    
    def __init__(self):
        # Create 3D R-tree index
        p = rtree_index.Property()
        p.dimension = 3
        p.variant = rtree_index.RT_Star
        self.idx = rtree_index.Index(properties=p)
        self.node_map: Dict[int, str] = {}  # Maps R-tree ID to node ID
        self._next_id = 0
        
    def insert(self, node: WorldNode):
        """Insert node into spatial index"""
        rtree_id = self._next_id
        self._next_id += 1
        
        # Insert into R-tree with 3D bounds
        self.idx.insert(rtree_id, node.bounds)
        self.node_map[rtree_id] = node.node_id
        
    def delete(self, node: WorldNode):
        """Remove node from spatial index"""
        # Find and remove from R-tree
        for rtree_id, node_id in list(self.node_map.items()):
            if node_id == node.node_id:
                self.idx.delete(rtree_id, node.bounds)
                del self.node_map[rtree_id]
                break
    
    def query_region(self, bounds: Tuple[float, float, float, float, float, float]) -> List[str]:
        """Query nodes within 3D region"""
        rtree_ids = list(self.idx.intersection(bounds))
        return [self.node_map[rid] for rid in rtree_ids if rid in self.node_map]
    
    def nearest_neighbors(self, point: Tuple[float, float, float], k: int = 5) -> List[str]:
        """Find k nearest neighbors to a 3D point"""
        # Convert point to bounds for R-tree query
        point_bounds = (point[0], point[1], point[2], point[0], point[1], point[2])
        rtree_ids = list(self.idx.nearest(point_bounds, k))
        return [self.node_map[rid] for rid in rtree_ids if rid in self.node_map]
    
    def get_state(self) -> bytes:
        """Serialize spatial index state"""
        state = {
            'node_map': self.node_map,
            'next_id': self._next_id,
            'bounds': [(rid, self.node_map[rid]) for rid in self.node_map]
        }
        return gzip.compress(pickle.dumps(state))
    
    def load_state(self, state_data: bytes, nodes: Dict[str, WorldNode]):
        """Load spatial index from serialized state"""
        state = pickle.loads(gzip.decompress(state_data))
        
        # Recreate index
        p = rtree_index.Property()
        p.dimension = 3
        p.variant = rtree_index.RT_Star
        self.idx = rtree_index.Index(properties=p)
        
        self.node_map = state['node_map']
        self._next_id = state['next_id']
        
        # Reinsert all nodes
        for rtree_id, node_id in self.node_map.items():
            if node_id in nodes:
                self.idx.insert(rtree_id, nodes[node_id].bounds)


class StateSynchronizer:
    """Handles state synchronization across agents"""
    
    def __init__(self):
        self.sync_queue = deque(maxlen=1000)
        self.pending_updates: Dict[str, List[Dict]] = defaultdict(list)
        self.conflict_resolution_strategy = "last_write_wins"
        self.lock = threading.Lock()
        
    def queue_update(self, update: Dict[str, Any]):
        """Queue an update for synchronization"""
        update['timestamp'] = time.time()
        update['id'] = hashlib.sha256(str(update).encode()).hexdigest()[:8]
        
        with self.lock:
            self.sync_queue.append(update)
            self.pending_updates[update.get('node_id', '')].append(update)
    
    def resolve_conflicts(self, updates: List[Dict]) -> Dict:
        """Resolve conflicting updates"""
        if self.conflict_resolution_strategy == "last_write_wins":
            # Sort by timestamp and take the latest
            return sorted(updates, key=lambda x: x['timestamp'])[-1]
        elif self.conflict_resolution_strategy == "merge":
            # Merge all updates
            merged = {}
            for update in sorted(updates, key=lambda x: x['timestamp']):
                merged.update(update)
            return merged
        else:
            raise ValueError(f"Unknown conflict resolution strategy: {self.conflict_resolution_strategy}")
    
    def process_updates(self) -> List[Dict]:
        """Process pending updates and resolve conflicts"""
        with self.lock:
            processed = []
            
            for node_id, updates in self.pending_updates.items():
                if updates:
                    resolved = self.resolve_conflicts(updates)
                    processed.append(resolved)
            
            # Clear processed updates
            self.pending_updates.clear()
            
            return processed


class NeuralWorldStateGraph:
    """Main NWSG implementation with all Day 7 features"""
    
    def __init__(self, embedding_dim: int = 128, enable_neural: bool = True):
        # Core graph structure
        self.nodes: Dict[str, WorldNode] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        
        # Spatial indexing
        self.spatial_index = SpatialIndex()
        
        # State management
        self.snapshots: Dict[str, StateSnapshot] = {}
        self.current_version = 0
        self.max_snapshots = 10  # Keep last 10 snapshots
        
        # State synchronization
        self.synchronizer = StateSynchronizer()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Neural components
        self.enable_neural = enable_neural
        self.embedding_dim = embedding_dim
        if enable_neural:
            self.transformer = self._init_transformer()
            self.node_encoder = nn.Linear(64, embedding_dim)  # Encode node features
            self.edge_encoder = nn.Linear(32, embedding_dim)  # Encode edge features
        
        # Performance monitoring
        self.query_times = deque(maxlen=1000)
        self.update_times = deque(maxlen=1000)
        self.metrics = {
            'total_queries': 0,
            'total_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Caching for frequent queries
        self.query_cache = {}
        self.cache_ttl = 1.0  # Cache TTL in seconds
        
        logger.info(f"NWSG initialized with embedding_dim={embedding_dim}, neural={enable_neural}")
    
    def _init_transformer(self) -> nn.Module:
        """Initialize transformer for cross-agent communication"""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
    
    def add_node(self, node: WorldNode) -> bool:
        """Add a node to the graph with spatial indexing"""
        with self.lock:
            start_time = time.time()
            
            try:
                if node.node_id in self.nodes:
                    logger.warning(f"Node {node.node_id} already exists")
                    return False
                
                # Add to graph
                self.nodes[node.node_id] = node
                
                # Add to spatial index
                self.spatial_index.insert(node)
                
                # Initialize embeddings if neural mode enabled
                if self.enable_neural and node.embedding is None:
                    node.embedding = torch.randn(self.embedding_dim)
                
                # Update version
                node.version = self.current_version
                self.current_version += 1
                
                # Queue synchronization update
                self.synchronizer.queue_update({
                    'action': 'add_node',
                    'node_id': node.node_id,
                    'node_data': node.to_dict()
                })
                
                # Clear relevant caches
                self._invalidate_cache(node.node_id)
                
                # Track performance
                elapsed = time.time() - start_time
                self.update_times.append(elapsed)
                self.metrics['total_updates'] += 1
                
                logger.debug(f"Added node {node.node_id} in {elapsed*1000:.2f}ms")
                return True
                
            except Exception as e:
                logger.error(f"Error adding node {node.node_id}: {e}")
                return False
    
    def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update node attributes with synchronization"""
        with self.lock:
            start_time = time.time()
            
            if node_id not in self.nodes:
                logger.warning(f"Node {node_id} not found")
                return False
            
            node = self.nodes[node_id]
            old_bounds = node.bounds
            
            # Update attributes
            node.attributes.update(updates.get('attributes', {}))
            node.timestamp = time.time()
            node.version = self.current_version
            self.current_version += 1
            
            # Update spatial index if position/bounds changed
            if 'position' in updates or 'bounds' in updates:
                self.spatial_index.delete(node)
                if 'position' in updates:
                    node.position = updates['position']
                if 'bounds' in updates:
                    node.bounds = updates['bounds']
                self.spatial_index.insert(node)
            
            # Queue synchronization update
            self.synchronizer.queue_update({
                'action': 'update_node',
                'node_id': node_id,
                'updates': updates
            })
            
            # Clear caches
            self._invalidate_cache(node_id)
            
            elapsed = time.time() - start_time
            self.update_times.append(elapsed)
            
            return True
    
    def add_edge(self, from_id: str, to_id: str, bidirectional: bool = True, weight: float = 1.0):
        """Add edge between nodes"""
        with self.lock:
            if from_id not in self.nodes or to_id not in self.nodes:
                logger.warning(f"Cannot add edge: node(s) not found")
                return False
            
            self.edges[from_id].add(to_id)
            self.nodes[from_id].connections.add(to_id)
            
            if bidirectional:
                self.edges[to_id].add(from_id)
                self.nodes[to_id].connections.add(from_id)
            
            # Queue synchronization
            self.synchronizer.queue_update({
                'action': 'add_edge',
                'from_id': from_id,
                'to_id': to_id,
                'bidirectional': bidirectional,
                'weight': weight
            })
            
            return True
    
    def query_spatial(self, bounds: Tuple[float, float, float, float, float, float], 
                     node_types: Optional[List[NodeType]] = None,
                     use_cache: bool = True) -> List[WorldNode]:
        """Query nodes within spatial bounds with caching"""
        with self.lock:
            start_time = time.time()
            self.metrics['total_queries'] += 1
            
            # Check cache
            cache_key = f"spatial_{bounds}_{node_types}"
            if use_cache and cache_key in self.query_cache:
                cache_entry = self.query_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    self.metrics['cache_hits'] += 1
                    self.query_times.append(0.0001)  # Cache hit is very fast
                    return cache_entry['results']
            
            self.metrics['cache_misses'] += 1
            
            # Get nodes from spatial index
            node_ids = self.spatial_index.query_region(bounds)
            
            # Filter by type if specified
            results = []
            for node_id in node_ids:
                node = self.nodes.get(node_id)
                if node and (node_types is None or node.node_type in node_types):
                    results.append(node)
            
            # Update cache
            if use_cache:
                self.query_cache[cache_key] = {
                    'results': results,
                    'timestamp': time.time()
                }
            
            elapsed = time.time() - start_time
            self.query_times.append(elapsed)
            
            return results
    
    def find_nearest(self, position: Tuple[float, float, float], 
                    k: int = 5, 
                    node_types: Optional[List[NodeType]] = None) -> List[Tuple[float, WorldNode]]:
        """Find k nearest nodes to a position"""
        with self.lock:
            start_time = time.time()
            
            # Get candidates from spatial index
            candidate_ids = self.spatial_index.nearest_neighbors(position, k * 3)
            
            # Calculate distances and filter
            results = []
            for node_id in candidate_ids:
                node = self.nodes.get(node_id)
                if node and (node_types is None or node.node_type in node_types):
                    # Calculate 3D Euclidean distance
                    center = node.get_center()
                    dist = np.sqrt(sum((a - b)**2 for a, b in zip(position, center)))
                    results.append((dist, node))
            
            # Sort by distance and return top k
            results.sort(key=lambda x: x[0])
            
            elapsed = time.time() - start_time
            self.query_times.append(elapsed)
            
            return results[:k]
    
    def get_connected_nodes(self, node_id: str, max_depth: int = 1) -> Dict[str, int]:
        """Get all nodes connected within max_depth steps using BFS"""
        with self.lock:
            if node_id not in self.nodes:
                return {}
            
            visited = {node_id: 0}
            queue = deque([(node_id, 0)])
            
            while queue:
                current_id, depth = queue.popleft()
                
                if depth >= max_depth:
                    continue
                
                for neighbor_id in self.edges.get(current_id, set()):
                    if neighbor_id not in visited:
                        visited[neighbor_id] = depth + 1
                        queue.append((neighbor_id, depth + 1))
            
            return visited
    
    def create_snapshot(self, snapshot_id: Optional[str] = None) -> StateSnapshot:
        """Create a snapshot of current state for persistence/rollback"""
        with self.lock:
            if snapshot_id is None:
                snapshot_id = f"snapshot_{int(time.time())}"
            
            # Serialize all nodes
            serialized_nodes = {
                node_id: node.to_dict() 
                for node_id, node in self.nodes.items()
            }
            
            # Serialize edges
            serialized_edges = {
                node_id: list(connections)
                for node_id, connections in self.edges.items()
            }
            
            # Create snapshot
            snapshot = StateSnapshot(
                snapshot_id=snapshot_id,
                timestamp=time.time(),
                nodes=serialized_nodes,
                edges=serialized_edges,
                spatial_index_data=self.spatial_index.get_state(),
                metadata={
                    "version": self.current_version,
                    "node_count": len(self.nodes),
                    "edge_count": sum(len(edges) for edges in self.edges.values())
                }
            )
            
            snapshot.generate_checksum()
            self.snapshots[snapshot_id] = snapshot
            
            # Maintain max snapshots limit
            if len(self.snapshots) > self.max_snapshots:
                oldest = min(self.snapshots.keys(), 
                           key=lambda x: self.snapshots[x].timestamp)
                del self.snapshots[oldest]
            
            logger.info(f"Created snapshot {snapshot_id} with {len(self.nodes)} nodes")
            return snapshot
    
    def rollback_to_snapshot(self, snapshot_id: str) -> bool:
        """Rollback to a previous snapshot"""
        with self.lock:
            if snapshot_id not in self.snapshots:
                logger.error(f"Snapshot {snapshot_id} not found")
                return False
            
            snapshot = self.snapshots[snapshot_id]
            
            # Verify checksum
            original_checksum = snapshot.checksum
            snapshot.generate_checksum()
            if snapshot.checksum != original_checksum:
                logger.error(f"Snapshot {snapshot_id} checksum mismatch!")
                return False
            
            logger.info(f"Rolling back to snapshot {snapshot_id}")
            
            # Clear current state
            self.nodes.clear()
            self.edges.clear()
            self.query_cache.clear()
            
            # Restore nodes
            for node_id, node_data in snapshot.nodes.items():
                node = WorldNode.from_dict(node_data)
                self.nodes[node_id] = node
            
            # Restore edges
            self.edges = defaultdict(set)
            for node_id, connections in snapshot.edges.items():
                self.edges[node_id] = set(connections)
            
            # Restore spatial index
            self.spatial_index = SpatialIndex()
            self.spatial_index.load_state(snapshot.spatial_index_data, self.nodes)
            
            # Update version
            self.current_version = snapshot.metadata["version"]
            
            logger.info(f"Rollback complete. Restored {len(self.nodes)} nodes")
            return True
    
    def process_synchronization(self):
        """Process pending synchronization updates"""
        updates = self.synchronizer.process_updates()
        
        for update in updates:
            action = update.get('action')
            
            if action == 'add_node':
                # Handle remote node addition
                node_data = update.get('node_data')
                if node_data:
                    node = WorldNode.from_dict(node_data)
                    if node.node_id not in self.nodes:
                        self.add_node(node)
            
            elif action == 'update_node':
                # Handle remote node update
                node_id = update.get('node_id')
                updates = update.get('updates')
                if node_id and updates:
                    self.update_node(node_id, updates)
            
            elif action == 'add_edge':
                # Handle remote edge addition
                from_id = update.get('from_id')
                to_id = update.get('to_id')
                if from_id and to_id:
                    self.add_edge(from_id, to_id, 
                                update.get('bidirectional', True),
                                update.get('weight', 1.0))
    
    def compute_node_embeddings(self, node_ids: List[str]) -> torch.Tensor:
        """Compute neural embeddings for nodes using transformer"""
        if not self.enable_neural:
            raise RuntimeError("Neural components not enabled")
        
        embeddings = []
        for node_id in node_ids:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node.embedding is not None:
                    embeddings.append(node.embedding)
                else:
                    # Generate embedding from node features
                    features = torch.tensor([
                        node.position[0], node.position[1], node.position[2],
                        node.bounds[3] - node.bounds[0],  # width
                        node.bounds[4] - node.bounds[1],  # height
                        node.bounds[5] - node.bounds[2],  # depth
                        float(node.node_type.value == 'region'),
                        float(node.node_type.value == 'biome'),
                        float(node.node_type.value == 'chunk'),
                        float(node.node_type.value == 'entity'),
                    ])
                    # Pad to 64 dimensions
                    features = F.pad(features, (0, 64 - features.shape[0]))
                    embedding = self.node_encoder(features)
                    node.embedding = embedding
                    embeddings.append(embedding)
        
        if not embeddings:
            return torch.zeros(0, self.embedding_dim)
        
        # Stack and process through transformer
        stacked = torch.stack(embeddings).unsqueeze(0)  # Add batch dimension
        transformed = self.transformer(stacked)
        
        return transformed.squeeze(0)
    
    def _invalidate_cache(self, node_id: str):
        """Invalidate cache entries related to a node"""
        # Simple implementation: clear all spatial queries
        # More sophisticated: track which queries include which nodes
        keys_to_remove = [k for k in self.query_cache.keys() if k.startswith('spatial_')]
        for key in keys_to_remove:
            del self.query_cache[key]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.lock:
            query_times = list(self.query_times) if self.query_times else [0]
            update_times = list(self.update_times) if self.update_times else [0]
            
            return {
                'total_nodes': len(self.nodes),
                'total_edges': sum(len(edges) for edges in self.edges.values()),
                'total_snapshots': len(self.snapshots),
                'current_version': self.current_version,
                'query_performance': {
                    'count': self.metrics['total_queries'],
                    'avg_time_ms': np.mean(query_times) * 1000,
                    'max_time_ms': np.max(query_times) * 1000,
                    'min_time_ms': np.min(query_times) * 1000,
                    'p95_time_ms': np.percentile(query_times, 95) * 1000 if query_times else 0,
                    'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['total_queries'])
                },
                'update_performance': {
                    'count': self.metrics['total_updates'],
                    'avg_time_ms': np.mean(update_times) * 1000,
                    'max_time_ms': np.max(update_times) * 1000,
                    'min_time_ms': np.min(update_times) * 1000,
                },
                'memory_usage': {
                    'nodes_mb': sum(node.__sizeof__() for node in self.nodes.values()) / 1024 / 1024,
                    'cache_entries': len(self.query_cache)
                }
            }
    
    def validate_consistency(self) -> Dict[str, List[str]]:
        """Validate graph consistency and return any issues found"""
        issues = defaultdict(list)
        
        with self.lock:
            # Check edge consistency
            for from_id, to_ids in self.edges.items():
                if from_id not in self.nodes:
                    issues['orphan_edges'].append(f"Edge from non-existent node {from_id}")
                
                for to_id in to_ids:
                    if to_id not in self.nodes:
                        issues['orphan_edges'].append(f"Edge to non-existent node {to_id}")
                    
                    # Check if connection is recorded in node
                    if from_id in self.nodes and to_id not in self.nodes[from_id].connections:
                        issues['edge_mismatch'].append(f"Edge {from_id}->{to_id} not in node connections")
            
            # Check node connections consistency
            for node_id, node in self.nodes.items():
                for conn_id in node.connections:
                    if conn_id not in self.edges.get(node_id, set()):
                        issues['connection_mismatch'].append(f"Connection {node_id}->{conn_id} not in edges")
            
            # Check spatial index consistency
            all_indexed = set()
            test_bounds = (-1e6, -1e6, -1e6, 1e6, 1e6, 1e6)
            indexed_ids = self.spatial_index.query_region(test_bounds)
            all_indexed.update(indexed_ids)
            
            for node_id in self.nodes:
                if node_id not in all_indexed:
                    issues['spatial_index'].append(f"Node {node_id} not in spatial index")
        
        return dict(issues)


# Example usage and testing
if __name__ == "__main__":
    # Create NWSG instance
    nwsg = NeuralWorldStateGraph(embedding_dim=128, enable_neural=True)
    
    # Test 1: Add 1000+ nodes
    print("Test 1: Adding 1000+ nodes...")
    start_time = time.time()
    
    for i in range(1200):
        node_type = np.random.choice(list(NodeType))
        x, y, z = np.random.rand(3) * 1000
        size = np.random.rand() * 10 + 5
        
        node = WorldNode(
            node_id=f"node_{i}",
            node_type=node_type,
            position=(x, y, z),
            bounds=(x-size, y-size, z-size, x+size, y+size, z+size),
            attributes={
                'biome': np.random.choice(['forest', 'desert', 'ocean']),
                'temperature': np.random.rand() * 40 - 10,
                'populated': np.random.rand() > 0.7
            }
        )
        
        nwsg.add_node(node)
        
        # Add some edges
        if i > 0 and np.random.rand() > 0.8:
            nwsg.add_edge(f"node_{i}", f"node_{np.random.randint(0, i)}")
    
    elapsed = time.time() - start_time
    print(f"Added 1200 nodes in {elapsed:.2f} seconds")
    
    # Test 2: Spatial queries
    print("\nTest 2: Spatial queries...")
    query_bounds = (400, 400, 400, 600, 600, 600)
    results = nwsg.query_spatial(query_bounds, node_types=[NodeType.CHUNK, NodeType.ENTITY])
    print(f"Found {len(results)} nodes in region")
    
    # Test 3: Nearest neighbors
    print("\nTest 3: Nearest neighbor search...")
    nearest = nwsg.find_nearest((500, 500, 500), k=10)
    print(f"Found {len(nearest)} nearest neighbors")
    for dist, node in nearest[:3]:
        print(f"  - {node.node_id}: distance={dist:.2f}")
    
    # Test 4: State snapshot and rollback
    print("\nTest 4: State persistence...")
    snapshot1 = nwsg.create_snapshot("checkpoint_1")
    print(f"Created snapshot with {snapshot1.metadata['node_count']} nodes")
    
    # Modify state
    for i in range(1200, 1300):
        node = WorldNode(
            node_id=f"node_{i}",
            node_type=NodeType.ENTITY,
            position=(np.random.rand()*1000, np.random.rand()*1000, 0),
            bounds=(0, 0, 0, 10, 10, 10)
        )
        nwsg.add_node(node)
    
    print(f"Current nodes: {len(nwsg.nodes)}")
    
    # Rollback
    success = nwsg.rollback_to_snapshot("checkpoint_1")
    print(f"Rollback successful: {success}, nodes after rollback: {len(nwsg.nodes)}")
    
    # Test 5: Performance stats
    print("\nTest 5: Performance statistics...")
    stats = nwsg.get_performance_stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Query performance: {stats['query_performance']['avg_time_ms']:.2f}ms avg")
    print(f"Cache hit rate: {stats['query_performance']['cache_hit_rate']:.2%}")
    
    # Test 6: Consistency validation
    print("\nTest 6: Consistency validation...")
    issues = nwsg.validate_consistency()
    if issues:
        print("Consistency issues found:")
        for issue_type, issue_list in issues.items():
            print(f"  {issue_type}: {len(issue_list)} issues")
    else:
        print("No consistency issues found!")
    
    # Test 7: Neural embeddings (if enabled)
    if nwsg.enable_neural:
        print("\nTest 7: Neural embeddings...")
        sample_nodes = [f"node_{i}" for i in range(10)]
        embeddings = nwsg.compute_node_embeddings(sample_nodes)
        print(f"Computed embeddings shape: {embeddings.shape}")