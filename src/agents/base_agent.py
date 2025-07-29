"""
Base Agent class for MADWE multi-agent system
Day 8 - Production Code
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, List
import threading
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto

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
    custom_config: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents"""
    
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
        
        # Performance tracking
        self._processed_messages = 0
        self._processing_times: List[float] = []
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
        self._message_handlers[MessageType.STATE_UPDATE] = self.handle_state_update
        self._message_handlers[MessageType.RESPONSE] = self._handle_response
        self._message_handlers[MessageType.COORDINATION] = self.handle_coordination
    
    @abstractmethod
    def _initialize(self):
        """Initialize agent-specific components"""
        pass
    
    @abstractmethod
    def handle_command(self, message: Message):
        """Handle command messages"""
        pass
    
    @abstractmethod
    def handle_query(self, message: Message):
        """Handle query messages"""
        pass
    
    @abstractmethod
    def handle_state_update(self, message: Message):
        """Handle state update messages"""
        pass
    
    def handle_coordination(self, message: Message):
        """Handle multi-agent coordination messages"""
        coord_type = message.payload.get('coordination_type')
        if coord_type == 'sync':
            self._handle_sync_request(message)
    
    def _handle_response(self, message: Message):
        """Handle response messages"""
        if message.correlation_id and message.correlation_id in self._pending_responses:
            self._pending_responses[message.correlation_id] = message
    
    def _handle_sync_request(self, message: Message):
        """Handle synchronization requests"""
        sync_data = self.get_sync_data()
        self.send_message(
            message.sender_id,
            MessageType.RESPONSE,
            {
                'sync_data': sync_data,
                'timestamp': time.time()
            },
            correlation_id=message.message_id
        )
    
    def get_sync_data(self) -> Dict[str, Any]:
        """Get synchronization data - override in subclasses"""
        return {
            'agent_id': self.agent_id,
            'state': self.state.name,
            'processed_messages': self._processed_messages
        }
    
    def send_message(self, recipient_id: str, message_type: MessageType, 
                    payload: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL,
                    correlation_id: Optional[str] = None) -> bool:
        """Send message to another agent"""
        message = Message(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id
        )
        return self.router.route_message(message)
    
    def send_and_wait(self, recipient_id: str, message_type: MessageType,
                     payload: Dict[str, Any], timeout: float = 1.0) -> Optional[Message]:
        """Send message and wait for response"""
        correlation_id = str(time.time())
        message = Message(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id
        )
        
        # Clear any old response
        self._pending_responses.pop(correlation_id, None)
        
        # Send message
        if not self.router.route_message(message):
            return None
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            if correlation_id in self._pending_responses:
                return self._pending_responses.pop(correlation_id)
            time.sleep(0.01)
        
        return None
    
    def broadcast(self, payload: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL):
        """Broadcast message to all agents"""
        message = Message(
            sender_id=self.agent_id,
            message_type=MessageType.BROADCAST,
            priority=priority,
            payload=payload
        )
        return self.router.route_message(message)
    
    def publish_event(self, event_type: str, event_data: Dict[str, Any],
                     priority: MessagePriority = MessagePriority.NORMAL):
        """Publish an event"""
        message = Message(
            sender_id=self.agent_id,
            message_type=MessageType.EVENT,
            priority=priority,
            payload={
                'event_type': event_type,
                'event_data': event_data,
                'source_agent': self.agent_id
            }
        )
        return self.router.route_message(message)
    
    def subscribe_event(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        
        # Register with router
        self.router.subscribe_event(event_type, self.agent_id, handler)
    
    def start(self):
        """Start agent message processing"""
        if self.state not in [AgentState.READY, AgentState.STOPPED]:
            logger.warning(f"Cannot start agent in state: {self.state}")
            return
        
        self._running = True
        self.state = AgentState.RUNNING
        
        # Start sync processing thread
        self._thread = threading.Thread(target=self._process_messages, daemon=True)
        self._thread.start()
        
        logger.info(f"Agent {self.agent_id} started")
    
    def stop(self):
        """Stop agent gracefully"""
        if self.state != AgentState.RUNNING:
            return
        
        logger.info(f"Stopping agent {self.agent_id}")
        self.state = AgentState.STOPPING
        self._running = False
        
        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        # Cleanup
        self.router.unregister_agent(self.agent_id)
        self.state = AgentState.STOPPED
        logger.info(f"Agent {self.agent_id} stopped")
    
    def pause(self):
        """Pause agent processing"""
        self.state = AgentState.PAUSED
        logger.info(f"Agent {self.agent_id} paused")
    
    def resume(self):
        """Resume agent processing"""
        if self.state == AgentState.PAUSED:
            self.state = AgentState.RUNNING
            logger.info(f"Agent {self.agent_id} resumed")
    
    def _process_messages(self):
        """Main message processing loop"""
        while self._running:
            try:
                if self.state != AgentState.RUNNING:
                    time.sleep(0.1)
                    continue
                
                # Get next message
                message = self.router.get_message(
                    self.agent_id, 
                    timeout=self.config.message_timeout
                )
                
                if message:
                    start_time = time.time()
                    self._handle_message(message)
                    processing_time = time.time() - start_time
                    
                    # Track metrics
                    self._processed_messages += 1
                    self._processing_times.append(processing_time)
                    if len(self._processing_times) > 100:
                        self._processing_times.pop(0)
                
                # Send heartbeat periodically
                if time.time() - self._last_heartbeat > 10.0:
                    self._send_heartbeat()
                    
            except Exception as e:
                logger.error(f"Error processing message in {self.agent_id}: {e}", exc_info=True)
                self.state = AgentState.ERROR
    
    def _handle_message(self, message: Message):
        """Route message to appropriate handler"""
        handler = self._message_handlers.get(message.message_type)
        if handler:
            handler(message)
        else:
            logger.warning(f"No handler for message type: {message.message_type} in {self.agent_id}")
    
    def _send_heartbeat(self):
        """Send heartbeat message"""
        self.publish_event('agent_heartbeat', {
            'agent_id': self.agent_id,
            'state': self.state.name,
            'processed_messages': self._processed_messages,
            'avg_processing_time': sum(self._processing_times) / len(self._processing_times) if self._processing_times else 0
        }, priority=MessagePriority.LOW)
        self._last_heartbeat = time.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.config.agent_type,
            'state': self.state.name,
            'processed_messages': self._processed_messages,
            'avg_processing_time_ms': (sum(self._processing_times) / len(self._processing_times) * 1000) if self._processing_times else 0,
            'uptime_seconds': time.time() - self._last_heartbeat + 10
        }