"""
Agent Communication Protocols for MADWE
Day 8 - Production Code
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class RequestType(Enum):
    """Types of requests between agents"""
    GENERATE_ASSET = "generate_asset"
    QUERY_WORLD_STATE = "query_world_state"
    UPDATE_WORLD_STATE = "update_world_state"
    SYNCHRONIZE = "synchronize"
    PREDICT_PLAYER = "predict_player"
    VALIDATE_COHERENCE = "validate_coherence"
    GENERATE_TERRAIN = "generate_terrain"
    GENERATE_CHARACTER = "generate_character"
    GENERATE_NARRATIVE = "generate_narrative"


@dataclass
class WorldStateQuery:
    """Query for world state information"""
    query_type: str
    position: Optional[Tuple[float, float, float]] = None
    radius: float = 50.0
    node_types: List[str] = field(default_factory=list)
    time_range: Optional[Tuple[float, float]] = None
    include_predictions: bool = False
    include_history: bool = False
    max_results: int = 100
    filter_criteria: Dict[str, Any] = field(default_factory=dict)
    
    def to_message_payload(self) -> Dict[str, Any]:
        return {
            'request_type': RequestType.QUERY_WORLD_STATE.value,
            'query': asdict(self)
        }


@dataclass
class AssetGenerationRequest:
    """Request for asset generation"""
    asset_type: str
    biome: str
    style_attributes: Dict[str, Any] = field(default_factory=dict)
    resolution: Tuple[int, int] = (512, 512)
    variations: int = 1
    lod_levels: List[int] = field(default_factory=lambda: [0])
    seamless: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    quality_preset: str = "balanced"
    use_cache: bool = True
    
    def to_message_payload(self) -> Dict[str, Any]:
        return {
            'request_type': RequestType.GENERATE_ASSET.value,
            'request': asdict(self)
        }


@dataclass
class CoherenceValidationRequest:
    """Request to validate coherence between generated content"""
    content_items: List[Dict[str, Any]]
    validation_type: str
    threshold: float = 0.8
    validation_rules: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_message_payload(self) -> Dict[str, Any]:
        return {
            'request_type': RequestType.VALIDATE_COHERENCE.value,
            'request': asdict(self)
        }


@dataclass
class PlayerPredictionRequest:
    """Request for player behavior prediction"""
    player_id: str
    player_history: List[Dict[str, Any]]
    prediction_horizon: int = 5
    confidence_threshold: float = 0.7
    include_branches: bool = True
    max_branches: int = 3
    prediction_types: List[str] = field(default_factory=lambda: ['movement', 'action'])
    context_window: int = 10
    
    def to_message_payload(self) -> Dict[str, Any]:
        return {
            'request_type': RequestType.PREDICT_PLAYER.value,
            'request': asdict(self)
        }


@dataclass
class TerrainGenerationRequest:
    """Request for terrain/chunk generation"""
    position: Tuple[int, int]
    size: Tuple[int, int] = (32, 32)
    biome: str = "forest"
    seed: Optional[int] = None
    detail_level: int = 1
    include_entities: bool = True
    seamless_edges: Dict[str, str] = field(default_factory=dict)
    
    def to_message_payload(self) -> Dict[str, Any]:
        return {
            'request_type': RequestType.GENERATE_TERRAIN.value,
            'request': asdict(self)
        }


@dataclass
class CharacterGenerationRequest:
    """Request for character generation"""
    character_type: str
    race: str
    class_type: str
    pose: str = "idle"
    equipment: List[str] = field(default_factory=list)
    style_attributes: Dict[str, Any] = field(default_factory=dict)
    biome_context: str = "neutral"
    variations: int = 1
    
    def to_message_payload(self) -> Dict[str, Any]:
        return {
            'request_type': RequestType.GENERATE_CHARACTER.value,
            'request': asdict(self)
        }


@dataclass
class NarrativeGenerationRequest:
    """Request for narrative/dialogue generation"""
    narrative_type: str
    context: Dict[str, Any]
    character_id: Optional[str] = None
    personality_traits: List[str] = field(default_factory=list)
    emotional_state: str = "neutral"
    dialogue_history: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    def to_message_payload(self) -> Dict[str, Any]:
        return {
            'request_type': RequestType.GENERATE_NARRATIVE.value,
            'request': asdict(self)
        }


class ProtocolValidator:
    """Validates message protocols and payloads"""
    
    @staticmethod
    def validate_request(request_type: RequestType, payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate request payload"""
        validators = {
            RequestType.GENERATE_ASSET: ProtocolValidator._validate_asset_request,
            RequestType.QUERY_WORLD_STATE: ProtocolValidator._validate_world_query,
            RequestType.PREDICT_PLAYER: ProtocolValidator._validate_player_prediction,
            RequestType.GENERATE_TERRAIN: ProtocolValidator._validate_terrain_request,
            RequestType.GENERATE_CHARACTER: ProtocolValidator._validate_character_request,
            RequestType.GENERATE_NARRATIVE: ProtocolValidator._validate_narrative_request
        }
        
        validator = validators.get(request_type)
        if not validator:
            return True, None
        
        return validator(payload)
    
    @staticmethod
    def _validate_asset_request(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        required = ['asset_type', 'biome']
        request = payload.get('request', {})
        for field in required:
            if field not in request:
                return False, f"Missing required field: {field}"
        
        asset_type = request['asset_type']
        valid_types = ['texture', 'sprite', 'character', 'effect', 'model']
        if asset_type not in valid_types:
            return False, f"Invalid asset_type: {asset_type}"
        
        return True, None
    
    @staticmethod
    def _validate_world_query(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        query = payload.get('query', {})
        if 'query_type' not in query:
            return False, "Missing query_type"
        
        valid_types = ['local', 'hierarchical', 'temporal', 'predictive']
        if query['query_type'] not in valid_types:
            return False, f"Invalid query_type: {query['query_type']}"
        
        return True, None
    
    @staticmethod
    def _validate_player_prediction(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        request = payload.get('request', {})
        if 'player_id' not in request:
            return False, "Missing player_id"
        
        if 'player_history' not in request or not isinstance(request['player_history'], list):
            return False, "Invalid or missing player_history"
        
        return True, None
    
    @staticmethod
    def _validate_terrain_request(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        request = payload.get('request', {})
        if 'position' not in request:
            return False, "Missing position"
        
        if not isinstance(request['position'], (list, tuple)) or len(request['position']) != 2:
            return False, "Invalid position format"
        
        return True, None
    
    @staticmethod
    def _validate_character_request(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        request = payload.get('request', {})
        required = ['character_type', 'race', 'class_type']
        for field in required:
            if field not in request:
                return False, f"Missing required field: {field}"
        
        return True, None
    
    @staticmethod
    def _validate_narrative_request(payload: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        request = payload.get('request', {})
        if 'narrative_type' not in request:
            return False, "Missing narrative_type"
        
        valid_types = ['dialogue', 'quest', 'lore', 'description']
        if request['narrative_type'] not in valid_types:
            return False, f"Invalid narrative_type: {request['narrative_type']}"
        
        return True, None