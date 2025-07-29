"""
Environment generation agent for MADWE
Day 8: Agent Communication - No Unity dependencies
"""

import time
from typing import Tuple, Dict, Any, List, Optional
import logging
import numpy as np
from dataclasses import dataclass
import json

from .base_agent import BaseAgent, AgentConfig
from unity_bridge.communication import Message, MessageType, MessagePriority, MessageRouter
from .protocols import TerrainGenerationRequest, RequestType

logger = logging.getLogger(__name__)


@dataclass 
class ChunkData:
    """Data structure for world chunks"""
    position: Tuple[int, int]
    biome: str
    tiles: List[List[int]]
    entities: List[Dict[str, Any]]
    generated_at: float
    version: int = 1


class EnvironmentAgent(BaseAgent):
    """Environment generation agent - manages world state and terrain generation"""
    
    def __init__(self, config: AgentConfig, router: MessageRouter):
        # Enable Unity compatibility for future integration
        config.unity_compatible = True
        super().__init__(config, router)
        
        # World state management
        self.world_state: Dict[str, ChunkData] = {}
        self.active_regions: Dict[str, Dict[str, Any]] = {}
        self.generation_queue = []
        
        # Performance tracking
        self.chunks_generated = 0
        self.generation_times = []
        
    def _initialize(self):
        """Initialize environment agent"""
        # Subscribe to relevant events
        self.router.subscribe_event("player_moved", self.agent_id, self.on_player_moved)
        self.router.subscribe_event("chunk_request", self.agent_id, self.on_chunk_request)
        self.router.subscribe_event("biome_transition", self.agent_id, self.on_biome_transition)
        
        logger.info(f"Environment agent {self.agent_id} initialized")
    
    def handle_command(self, message: Message):
        """Handle generation commands"""
        command = message.payload.get('command')
        
        if command == 'generate_chunk':
            chunk_pos = message.payload.get('position')
            biome = message.payload.get('biome', 'forest')
            self.generate_chunk(chunk_pos, biome)
            
        elif command == 'generate_region':
            region_pos = message.payload.get('position')
            biome_type = message.payload.get('biome_type')
            size = message.payload.get('size', (4, 4))
            self.generate_region(region_pos, biome_type, size)
            
        elif command == 'clear_cache':
            area = message.payload.get('area')
            self._clear_world_cache(area)
    
    def handle_query(self, message: Message):
        """Handle world state queries"""
        query_type = message.payload.get('query_type')
        
        if query_type == 'get_chunk':
            chunk_pos = tuple(message.payload.get('position'))
            chunk_data = self.get_chunk_data(chunk_pos)
            
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                {
                    'chunk_data': chunk_data.to_dict() if chunk_data else None,
                    'position': chunk_pos
                },
                correlation_id=message.message_id
            )
            
        elif query_type == 'get_region_info':
            region_pos = tuple(message.payload.get('position'))
            region_info = self.get_region_info(region_pos)
            
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                {'region_info': region_info},
                correlation_id=message.message_id
            )
            
        elif query_type == 'world_stats':
            stats = self.get_world_statistics()
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                {'stats': stats},
                correlation_id=message.message_id
            )
    
    def handle_state_update(self, message: Message):
        """Handle state updates from other agents"""
        update_type = message.payload.get('update_type')
        
        if update_type == 'chunk_modified':
            chunk_pos = tuple(message.payload.get('position'))
            modifications = message.payload.get('modifications')
            self._apply_chunk_modifications(chunk_pos, modifications)
            
        elif update_type == 'entity_spawned':
            chunk_pos = tuple(message.payload.get('chunk_position'))
            entity_data = message.payload.get('entity')
            self._add_entity_to_chunk(chunk_pos, entity_data)
    
    def handle_unity_request(self, message: Message):
        """Handle Unity-specific requests when Unity bridge is available"""
        request_type = message.payload.get('request_type')
        
        if request_type == 'immediate_chunk':
            # High-priority chunk generation for Unity
            position = tuple(message.payload.get('position'))
            self._generate_chunk_immediate(position, message)
            
        elif request_type == 'chunk_batch':
            # Batch chunk generation
            positions = message.payload.get('positions', [])
            self._handle_batch_generation(positions, message)
    
    def generate_chunk(self, position: Tuple[int, int], biome: str = None) -> ChunkData:
        """Generate a new chunk at the specified position"""
        start_time = time.time()
        
        # Check if chunk already exists
        chunk_key = f"{position[0]}_{position[1]}"
        if chunk_key in self.world_state:
            logger.info(f"Chunk at {position} already exists")
            return self.world_state[chunk_key]
        
        # Determine biome if not specified
        if biome is None:
            biome = self._determine_biome(position)
        
        # Generate chunk data
        chunk_data = ChunkData(
            position=position,
            biome=biome,
            tiles=self._generate_tiles(position, biome),
            entities=self._generate_entities(position, biome),
            generated_at=time.time()
        )
        
        # Store in world state
        self.world_state[chunk_key] = chunk_data
        self.chunks_generated += 1
        
        # Track performance
        generation_time = time.time() - start_time
        self.generation_times.append(generation_time)
        
        # Notify other agents
        self.publish_event('chunk_generated', {
            'position': position,
            'biome': biome,
            'chunk_key': chunk_key,
            'generation_time': generation_time
        })
        
        # Request assets for this chunk
        self._request_chunk_assets(chunk_data)
        
        logger.info(f"Generated chunk at {position} ({biome}) in {generation_time:.3f}s")
        
        return chunk_data
    
    def generate_region(self, position: Tuple[int, int], biome_type: str, 
                       size: Tuple[int, int] = (4, 4)):
        """Generate a larger region with consistent biome"""
        region_key = f"region_{position[0]}_{position[1]}"
        
        region_data = {
            'position': position,
            'biome_type': biome_type,
            'size': size,
            'chunks': [],
            'generated_at': time.time()
        }
        
        # Generate chunks within region
        for dx in range(size[0]):
            for dy in range(size[1]):
                chunk_pos = (position[0] * size[0] + dx, position[1] * size[1] + dy)
                chunk = self.generate_chunk(chunk_pos, biome_type)
                region_data['chunks'].append(chunk_pos)
        
        self.active_regions[region_key] = region_data
        
        # Notify completion
        self.publish_event('region_generated', {
            'position': position,
            'biome': biome_type,
            'chunk_count': size[0] * size[1]
        })
    
    def get_chunk_data(self, position: Tuple[int, int]) -> Optional[ChunkData]:
        """Get chunk data for a position"""
        chunk_key = f"{position[0]}_{position[1]}"
        return self.world_state.get(chunk_key)
    
    def get_region_info(self, position: Tuple[int, int]) -> Dict[str, Any]:
        """Get information about a region"""
        # Find chunks in region
        region_chunks = []
        region_size = 4  # Default region size
        
        for dx in range(region_size):
            for dy in range(region_size):
                chunk_pos = (position[0] * region_size + dx, position[1] * region_size + dy)
                chunk_key = f"{chunk_pos[0]}_{chunk_pos[1]}"
                if chunk_key in self.world_state:
                    region_chunks.append(chunk_pos)
        
        return {
            'position': position,
            'generated_chunks': len(region_chunks),
            'total_chunks': region_size * region_size,
            'chunks': region_chunks,
            'completeness': len(region_chunks) / (region_size * region_size)
        }
    
    def get_world_statistics(self) -> Dict[str, Any]:
        """Get world generation statistics"""
        biome_counts = {}
        entity_counts = {}
        
        for chunk in self.world_state.values():
            # Count biomes
            biome_counts[chunk.biome] = biome_counts.get(chunk.biome, 0) + 1
            
            # Count entities
            for entity in chunk.entities:
                entity_type = entity.get('type', 'unknown')
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        avg_gen_time = (sum(self.generation_times) / len(self.generation_times)) if self.generation_times else 0
        
        return {
            'total_chunks': len(self.world_state),
            'chunks_generated': self.chunks_generated,
            'active_regions': len(self.active_regions),
            'biome_distribution': biome_counts,
            'entity_distribution': entity_counts,
            'avg_generation_time': avg_gen_time,
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _generate_tiles(self, position: Tuple[int, int], biome: str) -> List[List[int]]:
        """Generate tile data for a chunk"""
        # Simple tile generation - in production, use WFC
        chunk_size = 32
        tiles = []
        
        # Generate based on biome
        tile_mappings = {
            'forest': [0, 0, 0, 1, 1, 2],  # grass, grass, grass, tree, tree, path
            'desert': [3, 3, 3, 3, 4, 5],  # sand variations, rock, oasis
            'cyberpunk': [6, 6, 7, 7, 8, 9],  # road, building, neon, plaza
            'dungeon': [10, 10, 10, 11, 12, 13]  # stone, wall, trap, treasure
        }
        
        biome_tiles = tile_mappings.get(biome, [0])
        
        for y in range(chunk_size):
            row = []
            for x in range(chunk_size):
                # Simple noise-based selection
                noise = (x * 7 + y * 13 + position[0] * 31 + position[1] * 37) % len(biome_tiles)
                row.append(biome_tiles[noise])
            tiles.append(row)
        
        return tiles
    
    def _generate_entities(self, position: Tuple[int, int], biome: str) -> List[Dict[str, Any]]:
        """Generate entities for a chunk"""
        entities = []
        
        # Entity templates by biome
        entity_templates = {
            'forest': [
                {'type': 'tree', 'density': 0.1},
                {'type': 'animal', 'density': 0.02},
                {'type': 'resource', 'subtype': 'herbs', 'density': 0.05}
            ],
            'desert': [
                {'type': 'cactus', 'density': 0.05},
                {'type': 'resource', 'subtype': 'minerals', 'density': 0.03}
            ],
            'cyberpunk': [
                {'type': 'npc', 'subtype': 'citizen', 'density': 0.08},
                {'type': 'vehicle', 'density': 0.04}
            ],
            'dungeon': [
                {'type': 'enemy', 'subtype': 'skeleton', 'density': 0.06},
                {'type': 'treasure', 'density': 0.02}
            ]
        }
        
        templates = entity_templates.get(biome, [])
        chunk_size = 32
        
        for template in templates:
            # Generate entities based on density
            count = int(chunk_size * chunk_size * template['density'])
            for _ in range(count):
                entity = {
                    'id': f"entity_{time.time()}_{np.random.randint(10000)}",
                    'type': template['type'],
                    'position': (
                        np.random.randint(0, chunk_size),
                        np.random.randint(0, chunk_size)
                    )
                }
                if 'subtype' in template:
                    entity['subtype'] = template['subtype']
                entities.append(entity)
        
        return entities
    
    def _determine_biome(self, position: Tuple[int, int]) -> str:
        """Determine biome based on position"""
        # Simple biome generation based on position
        biomes = ['forest', 'desert', 'cyberpunk', 'dungeon']
        
        # Use position to create biome regions
        biome_index = (position[0] // 10 + position[1] // 10) % len(biomes)
        return biomes[biome_index]
    
    def _request_chunk_assets(self, chunk_data: ChunkData):
        """Request assets for newly generated chunk"""
        # Send asset generation request
        self.send_message(
            "asset_agent",  # Assuming asset agent exists
            MessageType.GENERATE_CONTENT,
            {
                'request_type': 'chunk_assets',
                'chunk_position': chunk_data.position,
                'biome': chunk_data.biome,
                'tile_types': list(set(tile for row in chunk_data.tiles for tile in row)),
                'entity_types': list(set(e['type'] for e in chunk_data.entities))
            },
            priority=MessagePriority.NORMAL
        )
    
    def _apply_chunk_modifications(self, position: Tuple[int, int], 
                                  modifications: Dict[str, Any]):
        """Apply modifications to existing chunk"""
        chunk_key = f"{position[0]}_{position[1]}"
        if chunk_key in self.world_state:
            chunk = self.world_state[chunk_key]
            
            # Apply tile modifications
            if 'tiles' in modifications:
                for mod in modifications['tiles']:
                    x, y, new_tile = mod['x'], mod['y'], mod['tile']
                    if 0 <= x < len(chunk.tiles[0]) and 0 <= y < len(chunk.tiles):
                        chunk.tiles[y][x] = new_tile
            
            # Update version
            chunk.version += 1
            
            # Notify about modification
            self.publish_event('chunk_modified', {
                'position': position,
                'version': chunk.version
            })
    
    def _add_entity_to_chunk(self, position: Tuple[int, int], entity_data: Dict[str, Any]):
        """Add entity to existing chunk"""
        chunk_key = f"{position[0]}_{position[1]}"
        if chunk_key in self.world_state:
            chunk = self.world_state[chunk_key]
            chunk.entities.append(entity_data)
            chunk.version += 1
    
    def _clear_world_cache(self, area: Optional[Dict[str, Any]] = None):
        """Clear world cache for area or entire world"""
        if area is None:
            # Clear entire world
            cleared = len(self.world_state)
            self.world_state.clear()
            self.active_regions.clear()
            logger.info(f"Cleared entire world cache ({cleared} chunks)")
        else:
            # Clear specific area
            min_x = area.get('min_x', 0)
            min_y = area.get('min_y', 0)
            max_x = area.get('max_x', 100)
            max_y = area.get('max_y', 100)
            
            chunks_to_remove = []
            for chunk_key, chunk in self.world_state.items():
                if min_x <= chunk.position[0] <= max_x and min_y <= chunk.position[1] <= max_y:
                    chunks_to_remove.append(chunk_key)
            
            for chunk_key in chunks_to_remove:
                del self.world_state[chunk_key]
            
            logger.info(f"Cleared {len(chunks_to_remove)} chunks from cache")
    
    def _generate_chunk_immediate(self, position: Tuple[int, int], message: Message):
        """Generate chunk with highest priority for Unity"""
        # Generate immediately
        chunk = self.generate_chunk(position)
        
        # Send direct response
        self.send_message(
            message.sender_id,
            MessageType.UNITY_RESPONSE,
            {
                'chunk_data': chunk.to_dict(),
                'position': position
            },
            priority=MessagePriority.CRITICAL,
            correlation_id=message.message_id
        )
    
    def _handle_batch_generation(self, positions: List[Tuple[int, int]], message: Message):
        """Handle batch chunk generation request"""
        generated_chunks = []
        
        for pos in positions:
            chunk = self.generate_chunk(tuple(pos))
            generated_chunks.append({
                'position': pos,
                'chunk_key': f"{pos[0]}_{pos[1]}",
                'biome': chunk.biome
            })
        
        # Send batch response
        self.send_message(
            message.sender_id,
            MessageType.UNITY_RESPONSE,
            {
                'batch_result': generated_chunks,
                'count': len(generated_chunks)
            },
            priority=MessagePriority.HIGH,
            correlation_id=message.message_id
        )
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Rough estimate: each chunk ~10KB
        chunk_memory = len(self.world_state) * 0.01  # MB
        region_memory = len(self.active_regions) * 0.001  # MB
        return chunk_memory + region_memory
    
    def on_player_moved(self, message: Message):
        """Handle player movement events"""
        event_data = message.payload.get('event_data', {})
        new_position = event_data.get('position')
        
        if new_position:
            # Convert world position to chunk position
            chunk_pos = (int(new_position[0] // 32), int(new_position[1] // 32))
            
            # Check if chunk exists
            if not self.get_chunk_data(chunk_pos):
                logger.info(f"Player moved to ungenerated chunk {chunk_pos}, generating...")
                self.generate_chunk(chunk_pos)
            
            # Pre-generate surrounding chunks
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_pos = (chunk_pos[0] + dx, chunk_pos[1] + dy)
                    if not self.get_chunk_data(neighbor_pos):
                        # Queue for generation
                        self.generation_queue.append(neighbor_pos)
    
    def on_chunk_request(self, message: Message):
        """Handle chunk request events"""
        event_data = message.payload.get('event_data', {})
        position = tuple(event_data.get('position'))
        priority = event_data.get('priority', 'normal')
        requester = message.payload.get('source_agent')
        
        if priority == 'urgent' or not self.get_chunk_data(position):
            # Generate immediately
            chunk = self.generate_chunk(position)
            
            # Notify requester
            if requester:
                self.send_message(
                    requester,
                    MessageType.STATE_UPDATE,
                    {
                        'update_type': 'chunk_ready',
                        'position': position,
                        'chunk_data': chunk.to_dict()
                    }
                )
    
    def on_biome_transition(self, message: Message):
        """Handle biome transition events"""
        event_data = message.payload.get('event_data', {})
        from_biome = event_data.get('from_biome')
        to_biome = event_data.get('to_biome')
        transition_zone = event_data.get('transition_zone')
        
        logger.info(f"Handling biome transition: {from_biome} -> {to_biome}")
        
        # Could implement smooth transition generation here
        if transition_zone:
            # Generate transition chunks with mixed biome features
            pass


# Extension for ChunkData to support serialization
def chunk_data_to_dict(chunk: ChunkData) -> Dict[str, Any]:
    """Convert ChunkData to dictionary"""
    return {
        'position': chunk.position,
        'biome': chunk.biome,
        'tiles': chunk.tiles,
        'entities': chunk.entities,
        'generated_at': chunk.generated_at,
        'version': chunk.version
    }

# Add method to ChunkData
ChunkData.to_dict = chunk_data_to_dict