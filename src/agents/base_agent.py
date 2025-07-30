"""
Base Agent class for MADWE multi-agent system
Day 8 - Enhanced with event subscription and improved message handling
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, List, Set
import threading
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
import asyncio
from collections import deque

from unity_bridge.communication import (
    Message, MessageType, MessagePriority, MessageRouter
)

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class AgentConfig:
    """Configuration for agent initialization"""
    agent_id: str
    agent_type: str = "generic"
    queue_size: int = 1000
    message_timeout: float = 1.0
    enable_async: bool = False
    unity_compatible: bool = False
    custom_config: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Enhanced base class for all agents with event subscription"""
    
    def __init__(self, config: AgentConfig, router: MessageRouter):
        self.config = config
        self.agent_id = config.agent_id
        self.router = router
        
        # State management
        self.state = AgentState.INITIALIZING
        self._running = False
        self._thread = None
        
        # Message handling
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._pending_responses: Dict[str, Message] = {}
        self._response_lock = threading.Lock()
        
        # Event subscription tracking
        self._subscribed_events: Set[str] = set()
        self._event_patterns: List[str] = []
        
        # Performance tracking
        self._processed_messages = 0
        self._processing_times: deque = deque(maxlen=100)
        self._last_heartbeat = time.time()
        
        # Register with router
        self.router.register_agent(config.agent_id, config.queue_size)
        
        # Set up handlers
        self._setup_handlers()
        
        # Initialize agent-specific components
        self._initialize()
        
        self.state = AgentState.READY
        logger.info(f"Agent {self.agent_id} initialized (type: {config.agent_type})")
    
    def _setup_handlers(self):
        """Set up default message handlers"""
        self._message_handlers[MessageType.COMMAND] = self.handle_command
        self._message_handlers[MessageType.QUERY] = self.handle_query
        self._message_handlers[MessageType.RESPONSE] = self.handle_response
        self._message_handlers[MessageType.EVENT] = self.handle_event
        self._message_handlers[MessageType.STATE_UPDATE] = self.handle_state_update
        self._message_handlers[MessageType.COORDINATION] = self.handle_coordination
    
    @abstractmethod
    def _initialize(self):
        """Initialize agent-specific components"""
        pass
    
    def start(self):
        """Start the agent's message processing loop"""
        if self._running:
            return
            
        self._running = True
        self.state = AgentState.RUNNING
        self._thread = threading.Thread(target=self._process_messages, daemon=True)
        self._thread.start()
        logger.info(f"Agent {self.agent_id} started")
    
    def stop(self):
        """Stop the agent"""
        self._running = False
        self.state = AgentState.STOPPING
        
        if self._thread:
            self._thread.join(timeout=5.0)
            
        self.router.unregister_agent(self.agent_id)
        self.state = AgentState.STOPPED
        logger.info(f"Agent {self.agent_id} stopped")
    
    def _process_messages(self):
        """Main message processing loop"""
        while self._running:
            try:
                message = self.router.get_message(self.agent_id, timeout=0.1)
                
                if message:
                    start_time = time.time()
                    self._handle_message(message)
                    processing_time = time.time() - start_time
                    
                    self._processed_messages += 1
                    self._processing_times.append(processing_time)
                    
                # Send heartbeat periodically
                if time.time() - self._last_heartbeat > 30.0:
                    self._send_heartbeat()
                    
            except Exception as e:
                logger.error(f"Error processing message in {self.agent_id}: {e}")
                self.state = AgentState.ERROR
    
    def _handle_message(self, message: Message):
        """Handle incoming message"""
        handler = self._message_handlers.get(message.message_type)
        
        if handler:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Error in handler for {message.message_type}: {e}")
                
                # Send error response if query
                if message.message_type == MessageType.QUERY:
                    self.send_error_response(message, str(e))
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
    
    def subscribe_event(self, event_type: str, callback: Optional[Callable] = None):
        """Subscribe to an event type or pattern"""
        if callback is None:
            callback = self.handle_event
            
        self.router.subscribe_event(event_type, self.agent_id, callback)
        self._subscribed_events.add(event_type)
        
        if '*' in event_type:
            self._event_patterns.append(event_type)
            
        logger.debug(f"Agent {self.agent_id} subscribed to event: {event_type}")
    
    def unsubscribe_event(self, event_type: str):
        """Unsubscribe from an event"""
        # Note: Router doesn't have unsubscribe yet, track locally
        self._subscribed_events.discard(event_type)
        self._event_patterns = [p for p in self._event_patterns if p != event_type]
    
    def send_message(self, recipient_id: str, message_type: MessageType, 
                     payload: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL,
                     correlation_id: Optional[str] = None) -> Message:
        """Send a message to another agent"""
        message = Message(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id
        )
        
        self.router.route_message(message)
        return message
    
    def broadcast(self, payload: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL):
        """Broadcast a message to all agents"""
        message = Message(
            sender_id=self.agent_id,
            message_type=MessageType.BROADCAST,
            priority=priority,
            payload=payload
        )
        
        self.router.route_message(message)
    
    def emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event"""
        payload = {
            'event_type': event_type,
            'event_data': event_data,
            'timestamp': time.time()
        }
        
        message = Message(
            sender_id=self.agent_id,
            message_type=MessageType.EVENT,
            payload=payload
        )
        
        self.router.route_message(message)
    
    def send_and_wait(self, recipient_id: str, message_type: MessageType,
                      payload: Dict[str, Any], timeout: float = 5.0) -> Optional[Message]:
        """Send a message and wait for response"""
        message = self.send_message(recipient_id, message_type, payload)
        
        # Store pending response
        with self._response_lock:
            self._pending_responses[message.message_id] = None
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._response_lock:
                response = self._pending_responses.get(message.message_id)
                if response is not None:
                    del self._pending_responses[message.message_id]
                    return response
            time.sleep(0.01)
        
        # Timeout
        with self._response_lock:
            self._pending_responses.pop(message.message_id, None)
        return None
    
    def send_error_response(self, original_message: Message, error: str):
        """Send error response"""
        self.send_message(
            original_message.sender_id,
            MessageType.RESPONSE,
            {'error': error, 'success': False},
            correlation_id=original_message.message_id
        )
    
    def _send_heartbeat(self):
        """Send heartbeat event"""
        self.emit_event('agent_heartbeat', {
            'agent_id': self.agent_id,
            'state': self.state.name,
            'processed_messages': self._processed_messages,
            'avg_processing_time': sum(self._processing_times) / len(self._processing_times) if self._processing_times else 0
        })
        self._last_heartbeat = time.time()
    
    # Abstract methods for subclasses
    @abstractmethod
    def handle_command(self, message: Message):
        """Handle command messages"""
        pass
    
    @abstractmethod
    def handle_query(self, message: Message):
        """Handle query messages"""
        pass
    
    def handle_response(self, message: Message):
        """Handle response messages"""
        correlation_id = message.correlation_id
        if correlation_id:
            with self._response_lock:
                if correlation_id in self._pending_responses:
                    self._pending_responses[correlation_id] = message
    
    def handle_event(self, message: Message):
        """Handle event messages"""
        event_type = message.payload.get('event_type', 'unknown')
        logger.debug(f"Agent {self.agent_id} received event: {event_type}")
    
    def handle_state_update(self, message: Message):
        """Handle state update messages"""
        logger.debug(f"Agent {self.agent_id} received state update")
    
    def handle_coordination(self, message: Message):
        """Handle coordination messages"""
        logger.debug(f"Agent {self.agent_id} received coordination message")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'state': self.state.name,
            'processed_messages': self._processed_messages,
            'avg_processing_time': sum(self._processing_times) / len(self._processing_times) if self._processing_times else 0,
            'subscribed_events': list(self._subscribed_events),
            'uptime': time.time() - self._last_heartbeat
        }