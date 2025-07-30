"""
Agent Communication Protocols for MADWE
Day 8 - Defines message structures and request types
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple


class RequestType(Enum):
    """Types of generation requests"""
    TERRAIN_CHUNK = auto()
    ASSET_TEXTURE = auto()
    CHARACTER_MODEL = auto()
    NARRATIVE_EVENT = auto()
    BIOME_TRANSITION = auto()
    PLAYER_PREDICTION = auto()
    WORLD_STATE_UPDATE = auto()


@dataclass
class GenerationRequest:
    """Base class for generation requests"""
    request_id: str
    request_type: RequestType
    priority: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TerrainGenerationRequest(GenerationRequest):
    """Request for terrain/chunk generation"""
    position: Tuple[int, int] = (0, 0)
    size: Tuple[int, int] = (32, 32)
    biome: str = "forest"
    adjacent_chunks: List[Tuple[int, int]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.request_type = RequestType.TERRAIN_CHUNK


@dataclass
class AssetGenerationRequest(GenerationRequest):
    """Request for asset generation"""
    asset_type: str = "texture"  # texture, sprite, model
    style: str = "default"
    resolution: Tuple[int, int] = (512, 512)
    prompt: str = ""
    variations: int = 1
    
    def __post_init__(self):
        self.request_type = RequestType.ASSET_TEXTURE


@dataclass
class CoordinationMessage:
    """Message for multi-agent coordination"""
    coordinator_id: str
    participants: List[str]
    action: str
    parameters: Dict[str, Any]
    consensus_required: bool = False
    timeout: float = 30.0


@dataclass
class StateUpdateMessage:
    """Message for world state updates"""
    update_type: str
    node_id: str
    changes: Dict[str, Any]
    version: int
    timestamp: float


@dataclass
class PredictionRequest:
    """Request for player behavior prediction"""
    player_id: str
    current_position: Tuple[float, float, float]
    recent_actions: List[Dict[str, Any]]
    time_horizon: float  # seconds to predict ahead
    confidence_threshold: float = 0.7