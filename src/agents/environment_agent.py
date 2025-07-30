"""
Environment generation agent for MADWE
Day 8: Agent Communication - Fixed imports and enhanced with message queue system
"""

import time
from typing import Tuple, Dict, Any, List, Optional, Set
import logging
import numpy as np
from dataclasses import dataclass
import json
from collections import defaultdict
from pathlib import Path

from .base_agent import BaseAgent, AgentConfig
from unity_bridge.communication import Message, MessageType, MessagePriority, MessageRouter
from .protocols import TerrainGenerationRequest, RequestType, StateUpdateMessage
from wfc.hierarchical_wfc import HierarchicalWFC, BiomeTileRules
from models.nwsg.graph_network import NeuralWorldStateGraph, WorldNode, NodeType

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
    neighbors: Dict[str, Optional[Tuple[int, int]]] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = {'N': None, 'E': None, 'S': None, 'W': None}


class EnvironmentAgent(BaseAgent):
    """Environment generation agent - manages world state and terrain generation"""
    
    def __init__(self, config: AgentConfig, router: MessageRouter):
        # Enable Unity compatibility for future integration
        config.unity_compatible = True
        config.custom_config.update({
            'chunk_size': (32, 32),
            'world_bounds': (-1000, -1000, 1000, 1000),
            'biomes': ['forest', 'desert', 'snow', 'cyberpunk', 'dungeon'],
            'max_active_chunks': 100,
            'generation_radius': 3
        })
        super().__init__(config, router)
        
        # World state management
        self.world_state: Dict[str, ChunkData] = {}
        self.active_regions: Dict[str, Dict[str, Any]] = {}
        self.generation_queue: List[TerrainGenerationRequest] = []
        self.chunk_size = config.custom_config['chunk_size']
        
        # WFC generation
        self.hierarchical_wfc = None
        self.biome_rules = BiomeTileRules()
        
        # NWSG integration
        self.world_graph = None
        
        # Performance tracking
        self.chunks_generated = 0
        self.generation_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Coordination state
        self.pending_coordinations: Dict[str, Dict[str, Any]] = {}
        
    def _initialize(self):
        """Initialize environment agent"""
        # Subscribe to relevant events
        self.subscribe_event("player_moved")
        self.subscribe_event("chunk_request") 
        self.subscribe_event("biome_transition")
        self.subscribe_event("world_state_query")
        self.subscribe_event("agent_coordination.*")  # Pattern subscription
        
        # Initialize WFC
        try:
            config_path = Path("configs/generation/wfc_config.json")
            if config_path.exists():
                self.hierarchical_wfc = HierarchicalWFC(config_path)
            else:
                logger.warning("WFC config not found, using default rules")
                self._setup_default_wfc()
        except Exception as e:
            logger.error(f"Failed to initialize WFC: {e}")
            self._setup_default_wfc()
            
        # Initialize NWSG
        self.world_graph = NeuralWorldStateGraph()
        
        logger.info(f"Environment agent {self.agent_id} initialized")
    
    def _setup_default_wfc(self):
        """Setup default WFC without config file"""
        self.hierarchical_wfc = None  # Will use simple tile rules
    
    def handle_command(self, message: Message):
        """Handle generation commands"""
        command = message.payload.get('command')
        
        if command == 'generate_chunk':
            self._handle_generate_chunk(message)
            
        elif command == 'generate_region':
            self._handle_generate_region(message)
            
        elif command == 'clear_cache':
            self.world_state.clear()
            self.chunks_generated = 0
            
        elif command == 'set_biome':
            biome = message.payload.get('biome')
            if biome in self.config.custom_config['biomes']:
                self.current_biome = biome
                
    def handle_query(self, message: Message):
        """Handle world state queries"""
        query_type = message.payload.get('query_type')
        
        if query_type == 'chunk_exists':
            position = tuple(message.payload.get('position', (0, 0)))
            exists = self._chunk_key(position) in self.world_state
            
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                {'exists': exists, 'position': position},
                correlation_id=message.message_id
            )
            
        elif query_type == 'get_chunk':
            position = tuple(message.payload.get('position', (0, 0)))
            chunk_data = self._get_chunk_data(position)
            
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                {'chunk_data': chunk_data},
                correlation_id=message.message_id
            )
            
        elif query_type == 'world_stats':
            stats = self.get_world_stats()
            
            self.send_message(
                message.sender_id,
                MessageType.RESPONSE,
                stats,
                correlation_id=message.message_id
            )
    
    def handle_event(self, message: Message):
        """Handle events with priority routing"""
        event_type = message.payload.get('event_type')
        event_data = message.payload.get('event_data', {})
        
        if event_type == 'player_moved':
            self._on_player_moved(event_data)
            
        elif event_type == 'chunk_request':
            self._on_chunk_request(event_data)
            
        elif event_type == 'biome_transition':
            self._on_biome_transition(event_data)
            
        elif event_type.startswith('agent_coordination'):
            self._handle_coordination_event(message)
    
    def handle_coordination(self, message: Message):
        """Handle multi-agent coordination messages"""
        action = message.payload.get('action')
        
        if action == 'prepare_generation':
            # Coordinate with asset agent for texture generation
            self._coordinate_asset_generation(message)
            
        elif action == 'validate_coherence':
            # Validate world coherence across agents
            self._validate_coherence(message)
            
        elif action == 'consensus_request':
            # Participate in consensus protocol
            self._handle_consensus(message)
    
    def _handle_generate_chunk(self, message: Message):
        """Handle chunk generation request with priority"""
        position = tuple(message.payload.get('position', (0, 0)))
        biome = message.payload.get('biome', 'forest')
        priority = message.payload.get('priority', 2)
        
        # Create generation request
        request = TerrainGenerationRequest(
            request_id=f"chunk_{position[0]}_{position[1]}",
            position=position,
            size=self.chunk_size,
            biome=biome,
            priority=priority,
            adjacent_chunks=self._get_adjacent_positions(position)
        )
        
        # Add to priority queue
        self._enqueue_generation(request)
        
        # Process immediately if high priority
        if priority <= 1:
            self._process_generation_queue()
    
    def _enqueue_generation(self, request: TerrainGenerationRequest):
        """Add request to priority queue"""
        # Simple priority insertion
        inserted = False
        for i, existing in enumerate(self.generation_queue):
            if request.priority < existing.priority:
                self.generation_queue.insert(i, request)
                inserted = True
                break
                
        if not inserted:
            self.generation_queue.append(request)
    
    def _process_generation_queue(self):
        """Process pending generation requests"""
        if not self.generation_queue:
            return
            
        request = self.generation_queue.pop(0)
        
        # Check cache first
        chunk_key = self._chunk_key(request.position)
        if chunk_key in self.world_state:
            self.cache_hits += 1
            return
            
        self.cache_misses += 1
        
        # Generate chunk
        start_time = time.time()
        chunk_data = self._generate_chunk(request)
        generation_time = time.time() - start_time
        
        # Store chunk
        self.world_state[chunk_key] = chunk_data
        self.chunks_generated += 1
        self.generation_times.append(generation_time)
        
        # Update NWSG
        if self.world_graph:
            node = WorldNode(
                node_id=chunk_key,
                node_type=NodeType.CHUNK,
                position=(request.position[0], request.position[1], 0),
                bounds=(request.position[0], request.position[1], 
                       request.position[0] + self.chunk_size[0],
                       request.position[1] + self.chunk_size[1]),
                attributes={
                    'biome': chunk_data.biome,
                    'generated_at': chunk_data.generated_at,
                    'version': chunk_data.version
                }
            )
            self.world_graph.add_node(node)
        
        # Emit chunk generated event
        self.emit_event('chunk_generated', {
            'position': request.position,
            'biome': chunk_data.biome,
            'generation_time': generation_time
        })
        
        # Send to Unity if connected
        if self.config.unity_compatible:
            self._send_to_unity(chunk_data)
    
    def _generate_chunk(self, request: TerrainGenerationRequest) -> ChunkData:
        """Generate chunk using WFC or simple rules"""
        if self.hierarchical_wfc:
            # Use hierarchical WFC
            tile_map = self.hierarchical_wfc.generate(
                size=request.size,
                start_level='tile',
                constraints=request.constraints
            )
            tiles = tile_map.get('tile', np.zeros(request.size, dtype=int)).tolist()
        else:
            # Use simple biome rules
            tiles = self._generate_simple_tiles(request.biome, request.size)
        
        # Create chunk data
        chunk_data = ChunkData(
            position=request.position,
            biome=request.biome,
            tiles=tiles,
            entities=[],
            generated_at=time.time(),
            neighbors=self._get_neighbor_chunks(request.position)
        )
        
        return chunk_data
    
    def _generate_simple_tiles(self, biome: str, size: Tuple[int, int]) -> List[List[int]]:
        """Generate simple tile pattern for biome"""
        width, height = size
        tiles = []
        
        # Get biome-specific tiles
        if biome == 'forest':
            tile_weights = [(0, 3.0), (1, 1.0), (2, 0.5)]  # grass, tree, rock
        elif biome == 'desert':
            tile_weights = [(3, 4.0), (4, 0.5), (5, 0.2)]  # sand, cactus, rock
        else:
            tile_weights = [(0, 1.0)]  # default
        
        # Generate weighted random tiles
        tile_ids, weights = zip(*tile_weights)
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        for y in range(height):
            row = []
            for x in range(width):
                tile = np.random.choice(tile_ids, p=weights)
                row.append(int(tile))
            tiles.append(row)
            
        return tiles
    
    def _coordinate_asset_generation(self, message: Message):
        """Coordinate with asset agent for textures"""
        chunk_position = message.payload.get('chunk_position')
        biome = message.payload.get('biome')
        
        # Request texture generation from asset agent
        self.send_message(
            'asset_agent',
            MessageType.COMMAND,
            {
                'command': 'generate_biome_textures',
                'biome': biome,
                'chunk_position': chunk_position,
                'variations': 3
            },
            priority=MessagePriority.HIGH
        )
        
        # Store coordination state
        coord_id = f"asset_coord_{chunk_position}"
        self.pending_coordinations[coord_id] = {
            'start_time': time.time(),
            'chunk_position': chunk_position,
            'status': 'pending'
        }
    
    def _on_player_moved(self, event_data: Dict[str, Any]):
        """Handle player movement - generate nearby chunks"""
        player_pos = event_data.get('position', (0, 0))
        player_chunk = self._world_to_chunk(player_pos)
        
        # Generate chunks in radius
        radius = self.config.custom_config['generation_radius']
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                chunk_pos = (player_chunk[0] + dx, player_chunk[1] + dy)
                
                if self._chunk_key(chunk_pos) not in self.world_state:
                    # Prioritize based on distance
                    distance = abs(dx) + abs(dy)
                    priority = min(distance, 4)
                    
                    self.send_message(
                        self.agent_id,
                        MessageType.COMMAND,
                        {
                            'command': 'generate_chunk',
                            'position': chunk_pos,
                            'priority': priority
                        }
                    )
    
    def _chunk_key(self, position: Tuple[int, int]) -> str:
        """Generate unique key for chunk position"""
        return f"chunk_{position[0]}_{position[1]}"
    
    def _world_to_chunk(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world position to chunk coordinates"""
        chunk_x = int(world_pos[0] // self.chunk_size[0])
        chunk_y = int(world_pos[1] // self.chunk_size[1])
        return (chunk_x, chunk_y)
    
    def _get_adjacent_positions(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get adjacent chunk positions"""
        x, y = position
        return [
            (x, y-1),  # North
            (x+1, y),  # East
            (x, y+1),  # South
            (x-1, y),  # West
        ]
    
    def _get_neighbor_chunks(self, position: Tuple[int, int]) -> Dict[str, Optional[Tuple[int, int]]]:
        """Get neighbor chunk references"""
        x, y = position
        neighbors = {
            'N': (x, y-1) if self._chunk_key((x, y-1)) in self.world_state else None,
            'E': (x+1, y) if self._chunk_key((x+1, y)) in self.world_state else None,
            'S': (x, y+1) if self._chunk_key((x, y+1)) in self.world_state else None,
            'W': (x-1, y) if self._chunk_key((x-1, y)) in self.world_state else None,
        }
        return neighbors
    
    def _get_chunk_data(self, position: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Get chunk data as dictionary"""
        chunk_key = self._chunk_key(position)
        chunk = self.world_state.get(chunk_key)
        
        if chunk:
            return {
                'position': chunk.position,
                'biome': chunk.biome,
                'tiles': chunk.tiles,
                'entities': chunk.entities,
                'generated_at': chunk.generated_at,
                'version': chunk.version,
                'neighbors': chunk.neighbors
            }
        return None
    
    def _send_to_unity(self, chunk_data: ChunkData):
        """Placeholder for Unity communication"""
        # Will be implemented when Unity bridge is connected
        pass
    
    def _on_chunk_request(self, event_data: Dict[str, Any]):
        """Handle explicit chunk request"""
        position = tuple(event_data.get('position', (0, 0)))
        requester = event_data.get('requester_id')
        
        chunk_data = self._get_chunk_data(position)
        
        if chunk_data:
            # Send existing chunk
            self.send_message(
                requester,
                MessageType.RESPONSE,
                {'chunk_data': chunk_data}
            )
        else:
            # Generate new chunk
            self._handle_generate_chunk(Message(
                sender_id=requester,
                recipient_id=self.agent_id,
                message_type=MessageType.COMMAND,
                payload={
                    'command': 'generate_chunk',
                    'position': position,
                    'priority': 1
                }
            ))
    
    def _on_biome_transition(self, event_data: Dict[str, Any]):
        """Handle biome transition events"""
        from_biome = event_data.get('from_biome')
        to_biome = event_data.get('to_biome')
        transition_zone = event_data.get('transition_zone', [])
        
        logger.info(f"Biome transition: {from_biome} -> {to_biome}")
        
        # Update chunks in transition zone
        for chunk_pos in transition_zone:
            chunk_key = self._chunk_key(tuple(chunk_pos))
            if chunk_key in self.world_state:
                # Mark for regeneration with blended biome
                self.world_state[chunk_key].biome = f"{from_biome}_{to_biome}_blend"
    
    def _validate_coherence(self, message: Message):
        """Validate world coherence across chunks"""
        chunks_to_validate = message.payload.get('chunks', [])
        
        coherence_issues = []
        
        for chunk_pos in chunks_to_validate:
            chunk_key = self._chunk_key(tuple(chunk_pos))
            chunk = self.world_state.get(chunk_key)
            
            if not chunk:
                continue
                
            # Check neighbor consistency
            for direction, neighbor_pos in chunk.neighbors.items():
                if neighbor_pos:
                    neighbor_key = self._chunk_key(neighbor_pos)
                    neighbor = self.world_state.get(neighbor_key)
                    
                    if neighbor and neighbor.biome != chunk.biome:
                        # Check if valid transition
                        if not self._is_valid_biome_transition(chunk.biome, neighbor.biome):
                            coherence_issues.append({
                                'type': 'invalid_biome_transition',
                                'chunk': chunk_pos,
                                'neighbor': neighbor_pos,
                                'biomes': (chunk.biome, neighbor.biome)
                            })
        
        # Send validation results
        self.send_message(
            message.sender_id,
            MessageType.RESPONSE,
            {
                'valid': len(coherence_issues) == 0,
                'issues': coherence_issues
            },
            correlation_id=message.message_id
        )
    
    def _is_valid_biome_transition(self, biome1: str, biome2: str) -> bool:
        """Check if biome transition is valid"""
        valid_transitions = {
            'forest': ['desert', 'snow'],
            'desert': ['forest', 'cyberpunk'],
            'snow': ['forest', 'dungeon'],
            'cyberpunk': ['desert', 'dungeon'],
            'dungeon': ['snow', 'cyberpunk']
        }
        
        return (biome2 in valid_transitions.get(biome1, []) or 
                biome1 in valid_transitions.get(biome2, []) or
                '_blend' in biome1 or '_blend' in biome2)
    
    def _handle_coordination_event(self, message: Message):
        """Handle coordination events from other agents"""
        sub_type = message.payload.get('event_type').split('.')[-1]
        
        if sub_type == 'asset_ready':
            # Asset generation completed
            chunk_position = message.payload.get('event_data', {}).get('chunk_position')
            coord_id = f"asset_coord_{chunk_position}"
            
            if coord_id in self.pending_coordinations:
                self.pending_coordinations[coord_id]['status'] = 'completed'
                completion_time = time.time() - self.pending_coordinations[coord_id]['start_time']
                
                logger.info(f"Asset coordination completed for {chunk_position} in {completion_time:.2f}s")
                
                # Clean up
                del self.pending_coordinations[coord_id]
    
    def _handle_consensus(self, message: Message):
        """Participate in consensus protocol"""
        proposal = message.payload.get('proposal')
        proposal_type = proposal.get('type')
        
        vote = True  # Default accept
        
        if proposal_type == 'world_parameter_change':
            # Evaluate parameter change
            param = proposal.get('parameter')
            new_value = proposal.get('new_value')
            
            # Simple validation
            if param == 'chunk_size' and new_value != self.chunk_size:
                vote = False  # Don't change chunk size mid-generation
        
        # Send vote
        self.send_message(
            message.sender_id,
            MessageType.RESPONSE,
            {
                'vote': vote,
                'agent_id': self.agent_id,
                'reason': 'chunk_size_immutable' if not vote else 'accepted'
            },
            correlation_id=message.correlation_id
        )
    
    def get_world_stats(self) -> Dict[str, Any]:
        """Get world generation statistics"""
        return {
            'chunks_generated': self.chunks_generated,
            'chunks_cached': len(self.world_state),
            'active_regions': len(self.active_regions),
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0,
            'pending_generations': len(self.generation_queue),
            'pending_coordinations': len(self.pending_coordinations),
            'biomes_in_use': list(set(chunk.biome for chunk in self.world_state.values()))
        }