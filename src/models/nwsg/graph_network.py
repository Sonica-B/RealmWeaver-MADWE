"""
Neural World State Graph (NWSG) Implementation
Day 2: Architecture Design
Maintains global consistency across all generated content
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import json
import time

@dataclass
class WorldNode:
    """Represents a node in the world state graph"""
    node_id: str
    node_type: str  # 'region', 'biome', 'tile', 'entity'
    position: Tuple[float, float, float]  # 3D coordinates
    features: torch.Tensor
    metadata: Dict[str, Any]
    neighbors: List[str]
    timestamp: float
    
class SpatialIndex:
    """Efficient spatial indexing for world nodes"""
    
    def __init__(self, cell_size: float = 10.0):
        self.cell_size = cell_size
        self.cells = defaultdict(list)
        self.node_positions = {}
        
    def add_node(self, node_id: str, position: Tuple[float, float, float]):
        """Add node to spatial index"""
        cell = self._get_cell(position)
        self.cells[cell].append(node_id)
        self.node_positions[node_id] = position
        
    def _get_cell(self, position: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Get cell coordinates for position"""
        return (
            int(position[0] // self.cell_size),
            int(position[1] // self.cell_size),
            int(position[2] // self.cell_size)
        )
        
    def query_radius(self, position: Tuple[float, float, float], radius: float) -> List[str]:
        """Query nodes within radius of position"""
        results = []
        cells_to_check = self._get_nearby_cells(position, radius)
        
        for cell in cells_to_check:
            for node_id in self.cells.get(cell, []):
                node_pos = self.node_positions[node_id]
                if self._distance(position, node_pos) <= radius:
                    results.append(node_id)
                    
        return results
        
    def _get_nearby_cells(self, position: Tuple[float, float, float], radius: float) -> List[Tuple[int, int, int]]:
        """Get all cells that could contain nodes within radius"""
        center_cell = self._get_cell(position)
        cell_radius = int(np.ceil(radius / self.cell_size))
        
        cells = []
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                for dz in range(-cell_radius, cell_radius + 1):
                    cells.append((
                        center_cell[0] + dx,
                        center_cell[1] + dy,
                        center_cell[2] + dz
                    ))
        return cells
        
    def _distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between positions"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


class CrossModalAttention(nn.Module):
    """Multi-head attention for cross-modal consistency"""
    
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_len, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and final linear
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)
        
        return output


class NeuralWorldStateGraph(nn.Module):
    """Main NWSG implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Feature dimensions
        self.d_model = config.get('d_model', 512)
        self.d_agent = config.get('d_agent', 256)
        
        # Graph structure
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.spatial_index = SpatialIndex(cell_size=config.get('cell_size', 10.0))
        
        # Neural components
        self.cross_modal_attention = CrossModalAttention(self.d_model)
        
        # Agent feature encoders
        self.agent_encoders = nn.ModuleDict({
            'environment': nn.Linear(self.d_agent, self.d_model),
            'asset': nn.Linear(self.d_agent, self.d_model),
            'character': nn.Linear(self.d_agent, self.d_model),
            'narrative': nn.Linear(self.d_agent, self.d_model)
        })
        
        # Hierarchical encoders
        self.hierarchy_encoders = nn.ModuleDict({
            'region': nn.Linear(self.d_model, self.d_model),
            'biome': nn.Linear(self.d_model, self.d_model),
            'tile': nn.Linear(self.d_model, self.d_model),
            'entity': nn.Linear(self.d_model, self.d_model)
        })
        
        # Temporal buffer for maintaining continuity
        self.temporal_buffer = []
        self.max_temporal_length = config.get('max_temporal_length', 100)
        
        # Performance monitoring
        self.query_times = []
        
    def add_node(self, node: WorldNode):
        """Add node to the graph"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **node.metadata)
        self.spatial_index.add_node(node.node_id, node.position)
        
        # Add edges to neighbors
        for neighbor_id in node.neighbors:
            if neighbor_id in self.graph:
                self.graph.add_edge(node.node_id, neighbor_id)
                self.graph.add_edge(neighbor_id, node.node_id)
                
    def query_local_context(self, position: Tuple[float, float, float], 
                           radius: float = 50.0) -> Dict[str, Any]:
        """Query local context around position with sub-millisecond performance"""
        start_time = time.time()
        
        # Spatial query
        nearby_nodes = self.spatial_index.query_radius(position, radius)
        
        # Gather features
        if not nearby_nodes:
            return {'nodes': [], 'features': None, 'query_time': 0}
            
        features = []
        nodes_data = []
        
        for node_id in nearby_nodes:
            node = self.nodes[node_id]
            features.append(node.features)
            nodes_data.append({
                'id': node_id,
                'type': node.node_type,
                'position': node.position,
                'metadata': node.metadata
            })
            
        # Stack features
        features_tensor = torch.stack(features) if features else None
        
        query_time = (time.time() - start_time) * 1000  # milliseconds
        self.query_times.append(query_time)
        
        return {
            'nodes': nodes_data,
            'features': features_tensor,
            'query_time': query_time
        }
        
    def enforce_consistency(self, agent_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Enforce consistency across agent outputs using attention"""
        # Encode agent features
        encoded_features = []
        
        for agent_type, features in agent_outputs.items():
            if agent_type in self.agent_encoders:
                encoded = self.agent_encoders[agent_type](features)
                encoded_features.append(encoded)
                
        if not encoded_features:
            return torch.zeros(1, self.d_model)
            
        # Stack and apply cross-modal attention
        stacked_features = torch.stack(encoded_features, dim=1)
        
        # Self-attention for consistency
        consistent_features = self.cross_modal_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Average pool to get global state
        global_state = torch.mean(consistent_features, dim=1)
        
        return global_state
        
    def update_temporal_buffer(self, state: torch.Tensor, timestamp: float):
        """Update temporal buffer for continuity"""
        self.temporal_buffer.append({
            'state': state,
            'timestamp': timestamp
        })
        
        # Maintain buffer size
        if len(self.temporal_buffer) > self.max_temporal_length:
            self.temporal_buffer.pop(0)
            
    def get_hierarchical_context(self, node_id: str) -> Dict[str, torch.Tensor]:
        """Get hierarchical context for a node"""
        if node_id not in self.nodes:
            return {}
            
        node = self.nodes[node_id]
        context = {}
        
        # Get parent nodes in hierarchy
        ancestors = nx.ancestors(self.graph, node_id)
        
        for ancestor_id in ancestors:
            ancestor = self.nodes[ancestor_id]
            if ancestor.node_type in self.hierarchy_encoders:
                encoded = self.hierarchy_encoders[ancestor.node_type](ancestor.features)
                context[ancestor.node_type] = encoded
                
        return context
        
    def save_state(self, filepath: str):
        """Save graph state to file"""
        state = {
            'nodes': {
                node_id: {
                    'type': node.node_type,
                    'position': node.position,
                    'metadata': node.metadata,
                    'neighbors': node.neighbors,
                    'timestamp': node.timestamp
                }
                for node_id, node in self.nodes.items()
            },
            'graph_edges': list(self.graph.edges()),
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_state(self, filepath: str):
        """Load graph state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        # Rebuild graph
        self.graph.clear()
        self.nodes.clear()
        
        for node_id, node_data in state['nodes'].items():
            # Create dummy features
            features = torch.randn(self.d_model)
            
            node = WorldNode(
                node_id=node_id,
                node_type=node_data['type'],
                position=tuple(node_data['position']),
                features=features,
                metadata=node_data['metadata'],
                neighbors=node_data['neighbors'],
                timestamp=node_data['timestamp']
            )
            
            self.add_node(node)
            
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.query_times:
            return {'avg_query_time_ms': 0, 'max_query_time_ms': 0}
            
        return {
            'avg_query_time_ms': np.mean(self.query_times),
            'max_query_time_ms': np.max(self.query_times),
            'min_query_time_ms': np.min(self.query_times),
            'total_nodes': len(self.nodes),
            'total_edges': self.graph.number_of_edges()
        }


# Test the NWSG implementation
if __name__ == "__main__":
    config = {
        'd_model': 512,
        'd_agent': 256,
        'cell_size': 10.0,
        'max_temporal_length': 100
    }
    
    nwsg = NeuralWorldStateGraph(config)
    
    # Add test nodes
    for i in range(100):
        node = WorldNode(
            node_id=f"node_{i}",
            node_type="tile",
            position=(np.random.rand() * 100, np.random.rand() * 100, 0),
            features=torch.randn(512),
            metadata={'biome': 'forest'},
            neighbors=[],
            timestamp=time.time()
        )
        nwsg.add_node(node)
        
    # Test query performance
    results = nwsg.query_local_context((50, 50, 0), radius=20)
    print(f"Query found {len(results['nodes'])} nodes in {results['query_time']:.2f}ms")
    
    # Test consistency enforcement
    agent_outputs = {
        'environment': torch.randn(1, 256),
        'asset': torch.randn(1, 256),
        'character': torch.randn(1, 256)
    }
    
    consistent_state = nwsg.enforce_consistency(agent_outputs)
    print(f"Consistent state shape: {consistent_state.shape}")
    
    # Performance stats
    print(f"Performance stats: {nwsg.get_performance_stats()}")