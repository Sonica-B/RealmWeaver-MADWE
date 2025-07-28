"""
Agent Communication Protocols for MADWE
Day 2: Define message schemas and protocols
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json
import time
import uuid
from abc import ABC, abstractmethod


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 5
    LOW = 8
    BACKGROUND = 10


class RequestType(Enum):
    """Types of requests between agents"""
    GENERATE_ASSET = "generate_asset"
    QUERY_WORLD_STATE = "query_world_state"
    UPDATE_WORLD_STATE = "update_world_state"
    SYNCHRONIZE = "synchronize"
    PREDICT_PLAYER = "predict_player"
    VALIDATE_COHERENCE = "validate_coherence"


@dataclass
class AgentMessage:
    """Base message structure for inter-agent communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: str = ""
    priority: MessagePriority = MessagePriority.MEDIUM
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    correlation_id: Optional[str] = None
    timeout_ms: int = 50  # Default 50ms timeout
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        data = asdict(self)
        data['priority'] = self.priority.value
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentMessage':
        """Create message from JSON string"""
        data = json.loads(json_str)
        data['priority'] = MessagePriority(data['priority'])
        return cls(**data)


@dataclass
class WorldStateQuery:
    """Query for world state information"""
    query_type: str  # 'local', 'hierarchical', 'temporal'
    position: Optional[tuple[float, float, float]] = None
    radius: float = 50.0
    node_types: List[str] = field(default_factory=list)
    time_range: Optional[tuple[float, float]] = None
    include_predictions: bool = False
    
    
@dataclass
class AssetGenerationRequest:
    """Request for asset generation"""
    asset_type: str  # 'texture', 'sprite', 'character', 'effect'
    biome: str
    style_attributes: Dict[str, Any] = field(default_factory=dict)
    resolution: tuple[int, int] = (512, 512)
    variations: int = 1
    lod_levels: List[int] = field(default_factory=lambda: [0])
    seamless: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoherenceValidationRequest:
    """Request to validate coherence between generated content"""
    content_items: List[Dict[str, Any]]
    validation_type: str  # 'visual', 'semantic', 'narrative', 'gameplay'
    threshold: float = 0.8
    
    
@dataclass
class PlayerPredictionRequest:
    """Request for player behavior prediction"""
    player_history: List[Dict[str, Any]]
    prediction_horizon: int = 5  # seconds
    confidence_threshold: float = 0.7
    include_branches: bool = True
    max_branches: int = 3


class MessageHandler(ABC):
    """Abstract base class for message handlers"""
    
    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming message and optionally return response"""
        pass
    
    @abstractmethod
    def can_handle(self, message_type: str) -> bool:
        """Check if this handler can process the message type"""
        pass


class MessageRouter:
    """Routes messages between agents"""
    
    def __init__(self):
        self.handlers: Dict[str, List[MessageHandler]] = {}
        self.pending_responses: Dict[str, AgentMessage] = {}
        self.message_log: List[AgentMessage] = []
        self.max_log_size = 1000
        
    def register_handler(self, message_type: str, handler: MessageHandler):
        """Register a handler for a message type"""
        if message_type not in self.handlers:
            self.handlers[message_type] = []
        self.handlers[message_type].append(handler)
        
    async def route_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Route message to appropriate handler"""
        # Log message
        self.message_log.append(message)
        if len(self.message_log) > self.max_log_size:
            self.message_log.pop(0)
            
        # Find handlers
        handlers = self.handlers.get(message.message_type, [])
        
        # Route to first capable handler
        for handler in handlers:
            if handler.can_handle(message.message_type):
                response = await handler.handle_message(message)
                
                # Handle response tracking
                if message.requires_response and response:
                    response.correlation_id = message.message_id
                    
                return response
                
        return None
        
    def get_message_stats(self) -> Dict[str, Any]:
        """Get messaging statistics"""
        stats = {
            'total_messages': len(self.message_log),
            'messages_by_type': {},
            'average_latency': 0
        }
        
        # Count by type
        for msg in self.message_log:
            msg_type = msg.message_type
            stats['messages_by_type'][msg_type] = stats['messages_by_type'].get(msg_type, 0) + 1
            
        return stats


class MessageBuilder:
    """Helper class to build messages"""
    
    @staticmethod
    def create_world_query(sender_id: str, position: tuple[float, float, float], 
                          radius: float = 50.0) -> AgentMessage:
        """Create world state query message"""
        query = WorldStateQuery(
            query_type='local',
            position=position,
            radius=radius
        )
        
        return AgentMessage(
            sender_id=sender_id,
            recipient_id='nwsg',
            message_type=RequestType.QUERY_WORLD_STATE.value,
            priority=MessagePriority.HIGH,
            payload=asdict(query),
            requires_response=True,
            timeout_ms=30
        )
        
    @staticmethod
    def create_asset_request(sender_id: str, asset_type: str, biome: str,
                           **kwargs) -> AgentMessage:
        """Create asset generation request"""
        request = AssetGenerationRequest(
            asset_type=asset_type,
            biome=biome,
            **kwargs
        )
        
        return AgentMessage(
            sender_id=sender_id,
            recipient_id='asset_agent',
            message_type=RequestType.GENERATE_ASSET.value,
            priority=MessagePriority.MEDIUM,
            payload=asdict(request),
            requires_response=True,
            timeout_ms=100
        )
        
    @staticmethod
    def create_coherence_check(sender_id: str, content_items: List[Dict[str, Any]],
                             validation_type: str = 'visual') -> AgentMessage:
        """Create coherence validation request"""
        request = CoherenceValidationRequest(
            content_items=content_items,
            validation_type=validation_type
        )
        
        return AgentMessage(
            sender_id=sender_id,
            recipient_id='nwsg',
            message_type=RequestType.VALIDATE_COHERENCE.value,
            priority=MessagePriority.HIGH,
            payload=asdict(request),
            requires_response=True,
            timeout_ms=50
        )


# Protocol schemas for JSON serialization
PROTOCOL_SCHEMAS = {
    'agent_message': {
        'type': 'object',
        'properties': {
            'message_id': {'type': 'string'},
            'sender_id': {'type': 'string'},
            'recipient_id': {'type': 'string'},
            'message_type': {'type': 'string'},
            'priority': {'type': 'integer', 'minimum': 1, 'maximum': 10},
            'timestamp': {'type': 'number'},
            'payload': {'type': 'object'},
            'requires_response': {'type': 'boolean'},
            'correlation_id': {'type': ['string', 'null']},
            'timeout_ms': {'type': 'integer'}
        },
        'required': ['message_id', 'sender_id', 'message_type', 'timestamp']
    },
    
    'world_state_response': {
        'type': 'object',
        'properties': {
            'query_id': {'type': 'string'},
            'nodes': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'string'},
                        'type': {'type': 'string'},
                        'position': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 3,
                            'maxItems': 3
                        },
                        'metadata': {'type': 'object'}
                    }
                }
            },
            'query_time_ms': {'type': 'number'},
            'coherence_score': {'type': 'number', 'minimum': 0, 'maximum': 1}
        }
    },
    
    'asset_generation_response': {
        'type': 'object',
        'properties': {
            'request_id': {'type': 'string'},
            'asset_id': {'type': 'string'},
            'asset_paths': {
                'type': 'array',
                'items': {'type': 'string'}
            },
            'generation_time_ms': {'type': 'number'},
            'metadata': {'type': 'object'}
        }
    }
}


if __name__ == "__main__":
    # Test message creation and serialization
    msg = MessageBuilder.create_world_query("env_agent_01", (50, 50, 0), 30)
    print(f"Created message: {msg.to_json()}")
    
    # Test deserialization
    msg2 = AgentMessage.from_json(msg.to_json())
    print(f"Deserialized: {msg2}")
    
    # Test asset request
    asset_msg = MessageBuilder.create_asset_request(
        "env_agent_01",
        "texture",
        "forest",
        resolution=(512, 512),
        seamless=True
    )
    print(f"Asset request: {asset_msg.to_json()}")