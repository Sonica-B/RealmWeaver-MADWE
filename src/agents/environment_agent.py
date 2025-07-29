"""
Environment Generation Agent for MADWE
Day 5: Complete implementation for terrain generation
"""

import asyncio
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path
import numpy as np
import time
import json
from dataclasses import dataclass
from collections import defaultdict

from .base_agent import BaseAgent, Message, MessageType, AgentState
from ..wfc.hierarchical_wfc import HierarchicalWFC, WaveFunctionCollapse, BiomeTileRules
from ..unity_bridge.communication import UnityBridge, UnityMessage
from ..models.nwsg.graph_network import NeuralWorldStateGraph, WorldNode


@dataclass
class ChunkInfo:
    """Information about a generated chunk"""
    position: Tuple[int, int]
    biome: str
    tiles: np.ndarray
    timestamp: float
    generation_time: float
    metadata: Dict[str, Any] = None


class EnvironmentAgent(BaseAgent):
    """Agent responsible for environment generation using WFC"""
    
    def __init__(self, agent_id: str = "env_agent_01", 
                 config: Optional[Dict[str, Any]] = None,
                 message_bus = None):
        super().__init__(agent_id, "environment", config, message_bus)
        
        # Environment generation state
        self.chunks: Dict[Tuple[int, int], ChunkInfo] = {}
        self.active_chunks: Set[Tuple[int, int]] = set()
        self.generation_queue: asyncio.Queue = asyncio.Queue()
        
        # Biome configuration
        self.biomes = config.get('biomes', ['forest', 'desert', 'cyberpunk']) if config else ['forest', 'desert', 'cyberpunk']
        self.current_biome = 'forest'
        self.biome_transitions = self._setup_biome_transitions()
        
        # WFC setup
        self.chunk_size = config.get('chunk_size', (32, 32)) if config else (32, 32)
        self.wfc_configs = self._setup_wfc_configs()
        self.hierarchical_wfc = HierarchicalWFC()
        
        # Performance tracking
        self.generation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = config.get('max_cache_size', 100) if config else 100
        
        # External connections (set after creation)
        self.unity_bridge: Optional[UnityBridge] = None
        self.world_state_graph: Optional[NeuralWorldStateGraph] = None
        
        # Generation parameters
        self.generation_timeout = config.get('generation_timeout', 5.0) if config else 5.0
        self.max_concurrent_generations = config.get('max_concurrent', 3) if config else 3
        self.active_generations = 0
        
    async def _initialize(self):
        """Custom initialization for environment agent"""
        self.logger.info("Initializing environment agent components")
        
        # Pre-generate some common tile configurations
        for biome in self.biomes:
            self.wfc_configs[biome] = self._get_biome_tiles(biome)
            
        # Initialize chunk management
        self.chunk_cleanup_task = None
        
    def _setup_wfc_configs(self) -> Dict[str, List]:
        """Setup WFC configurations for each biome"""
        return {
            'forest': BiomeTileRules.get_forest_tiles(),
            'desert': BiomeTileRules.get_desert_tiles(),
            'cyberpunk': BiomeTileRules.get_cyberpunk_tiles()
        }
        
    def _get_biome_tiles(self, biome: str):
        """Get tiles for a specific biome"""
        if biome == 'forest':
            return BiomeTileRules.get_forest_tiles()
        elif biome == 'desert':
            return BiomeTileRules.get_desert_tiles()
        elif biome == 'cyberpunk':
            return BiomeTileRules.get_cyberpunk_tiles()
        else:
            return BiomeTileRules.get_forest_tiles()  # Default
            
    def _setup_biome_transitions(self) -> Dict[str, List[str]]:
        """Define valid biome transitions"""
        return {
            'forest': ['forest', 'desert'],
            'desert': ['desert', 'forest', 'cyberpunk'],
            'cyberpunk': ['cyberpunk', 'desert']
        }
        
    async def _run(self):
        """Main agent loop"""
        self.logger.info("Environment agent starting main loop")
        
        # Start chunk generation processor
        asyncio.create_task(self._process_generation_queue())
        
        # Start chunk cleanup task
        self.chunk_cleanup_task = asyncio.create_task(self._cleanup_chunks())
        
        while self.state == AgentState.READY:
            await asyncio.sleep(0.1)
            
    async def _process_message(self, message: Message):
        """Process incoming messages"""
        try:
            if message.type == MessageType.REQUEST:
                await self._handle_request(message)
            elif message.type == MessageType.COMMAND:
                await self._handle_command(message)
            elif message.type == MessageType.SYNC:
                await self._handle_sync(message)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            await self._send_error_response(message, str(e))
            
    async def _handle_request(self, message: Message):
        """Handle generation requests"""
        payload = message.payload
        request_type = payload.get('request_type')
        
        if request_type == 'generate_chunk':
            await self._queue_chunk_generation(
                position=tuple(payload['position']),
                biome=payload.get('biome', self.current_biome),
                priority=payload.get('priority', 5),
                requester=message.sender
            )
        elif request_type == 'query_chunk':
            await self._handle_chunk_query(message)
        elif request_type == 'get_stats':
            await self._send_stats(message.sender)
            
    async def _handle_command(self, message: Message):
        """Handle agent commands"""
        command = message.payload.get('command')
        
        if command == 'set_biome':
            self.current_biome = message.payload.get('biome', self.current_biome)
        elif command == 'clear_cache':
            self.chunks.clear()
            self.cache_hits = 0
            self.cache_misses = 0
        elif command == 'pause_generation':
            self.update_state(AgentState.PAUSED, "Generation paused")
        elif command == 'resume_generation':
            self.update_state(AgentState.READY, "Generation resumed")
            
    async def _queue_chunk_generation(self, position: Tuple[int, int], 
                                    biome: str, priority: int = 5,
                                    requester: str = None):
        """Queue a chunk for generation"""
        # Check cache first
        if position in self.chunks:
            self.cache_hits += 1
            await self._send_chunk_response(position, requester)
            return
            
        self.cache_misses += 1
        
        # Add to generation queue
        await self.generation_queue.put({
            'position': position,
            'biome': biome,
            'priority': priority,
            'requester': requester,
            'timestamp': time.time()
        })
        
    async def _process_generation_queue(self):
        """Process chunk generation requests"""
        while self.state != AgentState.SHUTDOWN:
            if self.state == AgentState.PAUSED:
                await asyncio.sleep(0.5)
                continue
                
            try:
                # Get next generation request
                request = await asyncio.wait_for(
                    self.generation_queue.get(), 
                    timeout=0.5
                )
                
                # Check if we can generate
                if self.active_generations >= self.max_concurrent_generations:
                    # Re-queue the request
                    await self.generation_queue.put(request)
                    await asyncio.sleep(0.1)
                    continue
                    
                # Generate chunk
                asyncio.create_task(self._generate_chunk_async(request))
                
            except asyncio.TimeoutError:
                continue
                
    async def _generate_chunk_async(self, request: Dict[str, Any]):
        """Generate a chunk asynchronously"""
        self.active_generations += 1
        start_time = time.time()
        
        position = request['position']
        biome = request['biome']
        requester = request.get('requester')
        
        try:
            self.update_state(AgentState.BUSY, f"Generating chunk at {position}")
            
            # Get tiles for biome
            tiles = self.wfc_configs.get(biome, self.wfc_configs['forest'])
            
            # Create WFC instance
            wfc = WaveFunctionCollapse(tiles, self.chunk_size)
            
            # Generate with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(wfc.collapse),
                timeout=self.generation_timeout
            )
            
            if result is None:
                raise ValueError("WFC generation failed")
                
            generation_time = time.time() - start_time
            
            # Create chunk info
            chunk = ChunkInfo(
                position=position,
                biome=biome,
                tiles=result,
                timestamp=time.time(),
                generation_time=generation_time,
                metadata={
                    'wfc_iterations': getattr(wfc, 'iterations', 0),
                    'tile_diversity': len(np.unique(result))
                }
            )
            
            # Store in cache
            self.chunks[position] = chunk
            self.active_chunks.add(position)
            
            # Track performance
            self.generation_times.append(generation_time)
            if len(self.generation_times) > 100:
                self.generation_times.pop(0)
                
            # Update world state graph if available
            if self.world_state_graph:
                await self._update_world_state(chunk)
                
            # Send to Unity if connected
            if self.unity_bridge and self.unity_bridge.connected:
                await self._send_to_unity(chunk)
                
            # Notify requester
            if requester:
                await self._send_chunk_response(position, requester)
                
            self.logger.info(f"Generated {biome} chunk at {position} in {generation_time:.2f}s")
            
        except asyncio.TimeoutError:
            self.logger.error(f"Generation timeout for chunk at {position}")
            if requester:
                await self._send_error_response(
                    None, 
                    f"Generation timeout for chunk at {position}",
                    requester
                )
        except Exception as e:
            self.logger.error(f"Generation error for chunk at {position}: {e}")
            if requester:
                await self._send_error_response(None, str(e), requester)
        finally:
            self.active_generations -= 1
            if self.active_generations == 0:
                self.update_state(AgentState.READY, "Generation complete")
                
    async def _send_chunk_response(self, position: Tuple[int, int], 
                                  recipient: str):
        """Send chunk data to requester"""
        if position not in self.chunks:
            return
            
        chunk = self.chunks[position]
        
        await self.send_message(
            recipient=recipient,
            message_type=MessageType.RESPONSE,
            payload={
                'response_type': 'chunk_data',
                'position': position,
                'biome': chunk.biome,
                'tiles': chunk.tiles.tolist(),
                'generation_time': chunk.generation_time,
                'metadata': chunk.metadata
            }
        )
        
    async def _send_error_response(self, original_message: Optional[Message], 
                                  error: str, recipient: str = None):
        """Send error response"""
        recipient = recipient or (original_message.sender if original_message else None)
        if not recipient:
            return
            
        await self.send_message(
            recipient=recipient,
            message_type=MessageType.ERROR,
            payload={
                'error': error,
                'original_request': original_message.payload if original_message else None
            }
        )
        
    async def _send_to_unity(self, chunk: ChunkInfo):
        """Send chunk data to Unity"""
        if not self.unity_bridge:
            return
            
        # Convert numpy array to Unity format
        tiles_data = []
        for y in range(chunk.tiles.shape[0]):
            for x in range(chunk.tiles.shape[1]):
                tile_id = int(chunk.tiles[y, x])
                tiles_data.append({
                    'x': chunk.position[0] + x,
                    'y': chunk.position[1] + y,
                    'tile_id': tile_id,
                    'biome': chunk.biome
                })
                
        message = UnityMessage(
            msg_type='chunk_generated',
            data={
                'position': chunk.position,
                'biome': chunk.biome,
                'tiles': tiles_data,
                'generation_time': chunk.generation_time
            }
        )
        
        self.unity_bridge.send_message(message)
        
    async def _update_world_state(self, chunk: ChunkInfo):
        """Update world state graph with new chunk"""
        if not self.world_state_graph:
            return
            
        # Create world node for chunk
        node = WorldNode(
            node_id=f"chunk_{chunk.position[0]}_{chunk.position[1]}",
            node_type="terrain_chunk",
            position=(chunk.position[0], 0, chunk.position[1]),
            data={
                'biome': chunk.biome,
                'tile_diversity': chunk.metadata.get('tile_diversity', 0),
                'generation_time': chunk.generation_time
            },
            timestamp=chunk.timestamp
        )
        
        await self.world_state_graph.add_node(node)
        
    async def _cleanup_chunks(self):
        """Periodically clean up old chunks"""
        while self.state != AgentState.SHUTDOWN:
            await asyncio.sleep(60)  # Every minute
            
            if len(self.chunks) > self.max_cache_size:
                # Remove oldest chunks
                sorted_chunks = sorted(
                    self.chunks.items(),
                    key=lambda x: x[1].timestamp
                )
                
                remove_count = len(self.chunks) - self.max_cache_size
                for position, _ in sorted_chunks[:remove_count]:
                    del self.chunks[position]
                    self.active_chunks.discard(position)
                    
                self.logger.info(f"Cleaned up {remove_count} old chunks")
                
    async def _send_stats(self, recipient: str):
        """Send agent statistics"""
        avg_gen_time = 0
        if self.generation_times:
            avg_gen_time = sum(self.generation_times) / len(self.generation_times)
            
        stats = {
            'cached_chunks': len(self.chunks),
            'active_chunks': len(self.active_chunks),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'avg_generation_time': avg_gen_time,
            'active_generations': self.active_generations,
            'queue_size': self.generation_queue.qsize()
        }
        
        await self.send_message(
            recipient=recipient,
            message_type=MessageType.RESPONSE,
            payload={
                'response_type': 'stats',
                'stats': stats
            }
        )
        
    async def generate_chunk(self, position: Tuple[int, int], 
                           biome: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Public method to generate a chunk"""
        biome = biome or self.current_biome
        
        # Queue generation
        await self._queue_chunk_generation(position, biome, priority=1)
        
        # Wait for generation (with timeout)
        timeout = 10
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if position in self.chunks:
                chunk = self.chunks[position]
                return {
                    'position': position,
                    'biome': chunk.biome,
                    'generation_time': chunk.generation_time,
                    'metadata': chunk.metadata
                }
            await asyncio.sleep(0.1)
            
        return None
        
    async def _cleanup(self):
        """Clean up resources"""
        if self.chunk_cleanup_task:
            self.chunk_cleanup_task.cancel()
            
        # Clear generation queue
        while not self.generation_queue.empty():
            self.generation_queue.get_nowait()