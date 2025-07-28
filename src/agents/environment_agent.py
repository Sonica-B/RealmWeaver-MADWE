"""
Environment Generation Agent for MADWE
Day 5: Friday, June 7 - Multi-Agent Foundation
Fixed version with missing imports
"""

import asyncio
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path
import numpy as np
import time
import json
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict  # FIXED: Added missing import

from agents.base_agent import BaseAgent, Message, MessageType, AgentState
from src.wfc.hierarchical_wfc import HierarchicalWFC, WaveFunctionCollapse, BiomeTileRules
from unity_bridge.communication import UnityBridge
from models.nwsg.graph_network import NeuralWorldStateGraph, WorldNode


@dataclass
class ChunkInfo:
    """Information about a generated chunk"""
    position: Tuple[int, int]
    biome: str
    tiles: np.ndarray
    timestamp: float
    generation_time: float


class EnvironmentAgent(BaseAgent):
    """Agent responsible for environment generation"""
    
    def __init__(self, agent_id: str, agent_type: str = "environment", 
                 config: Optional[Dict[str, Any]] = None, message_bus = None):
        super().__init__(agent_id, agent_type, config, message_bus)
        
        # Environment generation state
        self.chunks: Dict[Tuple[int, int], ChunkInfo] = {}
        self.active_chunks: Set[Tuple[int, int]] = set()
        self.generation_queue: List[Tuple[int, int]] = []
        
        # Biome configuration
        self.biomes = config.get('biomes', ['forest', 'desert', 'snow']) if config else ['forest', 'desert', 'snow']
        self.current_biome = 'forest'
        
        # WFC setup
        self.chunk_size = config.get('chunk_size', (32, 32)) if config else (32, 32)
        self.wfc_configs = self._setup_wfc_configs()
        
        # Performance tracking
        self.generation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Unity bridge and NWSG (set after creation)
        self.unity_bridge = None
        self.world_state_graph = None
        
    def _setup_wfc_configs(self) -> Dict[str, List]:
        """Setup WFC configurations for each biome"""
        return {
            'forest': BiomeTileRules.create_forest_tiles(),
            'desert': BiomeTileRules.create_desert_tiles(),
            'snow': BiomeTileRules.create_snow_tiles()
        }
        
    async def _initialize(self):
        """Initialize environment agent"""
        self.register_handler(MessageType.REQUEST, self._handle_generation_request)
        self.register_handler(MessageType.UPDATE, self._handle_player_update)
        
    async def _start(self):
        """Start environment generation tasks"""
        # Start chunk management loop
        chunk_task = asyncio.create_task(self._chunk_management_loop())
        self._tasks.append(chunk_task)
        
    async def _chunk_management_loop(self):
        """Manage chunk generation and cleanup"""
        while self._running:
            try:
                # Process generation queue
                if self.generation_queue:
                    position = self.generation_queue.pop(0)
                    await self.generate_chunk(position)
                    
                # Update world state
                await self._update_world_state()
                
                # Small delay
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Chunk management error: {e}")
                
    async def generate_chunk(self, position: Tuple[int, int]) -> Optional[ChunkInfo]:
        """Generate a chunk at the given position"""
        start_time = time.time()
        
        # Check cache
        if position in self.chunks:
            self.cache_hits += 1
            return self.chunks[position]
            
        self.cache_misses += 1
        
        # Get tiles for current biome
        tiles = self.wfc_configs.get(self.current_biome, self.wfc_configs['forest'])
        
        # Create WFC instance
        wfc = WaveFunctionCollapse(tiles, self.chunk_size)
        
        try:
            # Generate chunk
            result = wfc.collapse()
            
            # Create chunk info
            chunk = ChunkInfo(
                position=position,
                biome=self.current_biome,
                tiles=result,
                timestamp=time.time(),
                generation_time=(time.time() - start_time) * 1000  # ms
            )
            
            # Store in cache
            self.chunks[position] = chunk
            self.active_chunks.add(position)
            
            # Update world state graph if available
            if self.world_state_graph:
                await self._update_world_graph(chunk)
                
            # Send to Unity if connected
            if self.unity_bridge and self.unity_bridge.connected:
                self.unity_bridge.send_tile_update(result, position)
                
            self.logger.info(f"Generated {self.current_biome} chunk at {position} in {chunk.generation_time:.1f}ms")
            
            return {
                'position': position,
                'biome': self.current_biome,
                'tiles': result,
                'generation_time': chunk.generation_time
            }
            
        except Exception as e:
            self.logger.error(f"Chunk generation failed: {e}")
            return None
            
    async def _update_world_graph(self, chunk: ChunkInfo):
        """Update world state graph with new chunk"""
        if not self.world_state_graph:
            return
            
        # Create nodes for chunk
        base_x, base_y = chunk.position
        
        for y in range(chunk.tiles.shape[0]):
            for x in range(chunk.tiles.shape[1]):
                node_id = f"tile_{base_x + x}_{base_y + y}"
                
                node = WorldNode(
                    node_id=node_id,
                    node_type='tile',
                    position=(base_x + x, base_y + y, 0),
                    features=torch.randn(512),  # Placeholder features
                    metadata={
                        'biome': chunk.biome,
                        'tile_type': int(chunk.tiles[y, x])
                    },
                    neighbors=[],
                    timestamp=chunk.timestamp
                )
                
                self.world_state_graph.add_node(node)
                
    async def _handle_generation_request(self, message: Message):
        """Handle chunk generation requests"""
        position = message.payload.get('position', (0, 0))
        biome = message.payload.get('biome', self.current_biome)
        
        # Update biome if different
        if biome != self.current_biome:
            self.current_biome = biome
            
        # Add to generation queue
        self.generation_queue.append(position)
        
        # Send acknowledgment
        await self.send_message(
            Message(
                type=MessageType.RESPONSE,
                recipient=message.sender,
                correlation_id=message.id,
                payload={'status': 'queued', 'position': position}
            )
        )
        
    async def _handle_player_update(self, message: Message):
        """Handle player position updates"""
        player_pos = message.payload.get('position', (0, 0, 0))
        
        # Calculate chunks needed around player
        chunk_x = int(player_pos[0] // self.chunk_size[0])
        chunk_y = int(player_pos[1] // self.chunk_size[1])
        
        # Queue nearby chunks for generation
        radius = 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                chunk_pos = (chunk_x + dx, chunk_y + dy)
                if chunk_pos not in self.chunks and chunk_pos not in self.generation_queue:
                    self.generation_queue.append(chunk_pos)
                    
    async def _update_world_state(self):
        """Update world state statistics"""
        if not self.chunks:
            return
            
        # Calculate biome distribution
        biome_counts = defaultdict(int)  # Now properly imported
        for chunk in self.chunks.values():
            biome_counts[chunk.biome] += 1
            
        # Update metrics
        self.metrics['chunks_generated'] = len(self.chunks)
        self.metrics['active_chunks'] = len(self.active_chunks)
        self.metrics['cache_hit_rate'] = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        
        if self.generation_times:
            self.metrics['avg_generation_time'] = np.mean(self.generation_times[-20:])
            
    async def _pause(self):
        """Pause environment generation"""
        self.logger.info("Pausing environment generation")
        
    async def _resume(self):
        """Resume environment generation"""
        self.logger.info("Resuming environment generation")
        
    async def _shutdown(self):
        """Shutdown environment agent"""
        # Save chunk cache
        cache_path = Path("data/cache/chunks.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert chunks to serializable format
        chunk_data = {}
        for pos, chunk in self.chunks.items():
            chunk_data[f"{pos[0]}_{pos[1]}"] = {
                'position': pos,
                'biome': chunk.biome,
                'timestamp': chunk.timestamp
            }
            
        with open(cache_path, 'w') as f:
            json.dump(chunk_data, f)
            
        self.logger.info(f"Saved {len(self.chunks)} chunks to cache")
        
    async def _get_custom_status(self) -> Dict[str, Any]:
        """Get environment agent status"""
        return {
            'chunks_generated': len(self.chunks),
            'active_chunks': len(self.active_chunks),
            'queue_size': len(self.generation_queue),
            'current_biome': self.current_biome,
            'cache_stats': {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
            }
        }
        
    def _get_custom_state(self) -> Dict[str, Any]:
        """Get custom state for saving"""
        return {
            'current_biome': self.current_biome,
            'chunk_positions': list(self.chunks.keys()),
            'generation_times': self.generation_times[-100:]
        }
        
    def _load_custom_state(self, state: Dict[str, Any]):
        """Load custom state"""
        self.current_biome = state.get('current_biome', 'forest')
        self.generation_times = state.get('generation_times', [])


# Add required torch import at the top
import torch