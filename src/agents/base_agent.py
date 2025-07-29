"""
Base Agent Framework for MADWE
Day 5: Multi-Agent Foundation with state management
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime
import uuid
from collections import deque
import time


class MessageType(Enum):
    """Standard message types for inter-agent communication"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    STATUS_QUERY = "status_query"
    STATUS_RESPONSE = "status_response"
    ERROR = "error"
    SYNC = "sync"
    UPDATE = "update"
    COMMAND = "command"


class AgentState(Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class Message:
    """Standard message format for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    sender: str = ""
    recipient: Optional[str] = None  # None means broadcast
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    priority: int = 5  # 1-10, 1 being highest
    ttl: Optional[int] = None
    requires_ack: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "priority": self.priority,
            "ttl": self.ttl,
            "requires_ack": self.requires_ack
        }


class MessageBus:
    """Central message bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.agents: Dict[str, 'BaseAgent'] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_history: deque = deque(maxlen=1000)
        self.running = False
        
    async def register_agent(self, agent: 'BaseAgent'):
        """Register an agent with the message bus"""
        self.agents[agent.agent_id] = agent
        logging.info(f"Registered agent: {agent.agent_id}")
        
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logging.info(f"Unregistered agent: {agent_id}")
            
    async def publish(self, message: Message):
        """Publish a message to the bus"""
        await self.message_queue.put(message)
        self.message_history.append(message)
        
    async def subscribe(self, agent_id: str, callback: Callable):
        """Subscribe to messages"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
        
    async def start(self):
        """Start the message bus"""
        self.running = True
        asyncio.create_task(self._process_messages())
        
    async def _process_messages(self):
        """Process messages from the queue"""
        while self.running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=0.1
                )
                
                # Route message
                if message.recipient:
                    # Direct message
                    if message.recipient in self.subscribers:
                        for callback in self.subscribers[message.recipient]:
                            asyncio.create_task(callback(message))
                else:
                    # Broadcast
                    for agent_id, callbacks in self.subscribers.items():
                        if agent_id != message.sender:
                            for callback in callbacks:
                                asyncio.create_task(callback(message))
                                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Error processing message: {e}")


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""
    
    def __init__(self, agent_id: str, agent_type: str, 
                 config: Optional[Dict[str, Any]] = None,
                 message_bus: Optional[MessageBus] = None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.message_bus = message_bus
        
        # State management
        self.state = AgentState.INITIALIZING
        self.internal_state: Dict[str, Any] = {}
        self.state_history: deque = deque(maxlen=100)
        
        # Message handling
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.pending_requests: Dict[str, Message] = {}
        
        # Performance tracking
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "processing_times": deque(maxlen=100),
            "errors": 0
        }
        
        # Logger
        self.logger = logging.getLogger(f"{self.agent_type}.{self.agent_id}")
        
    async def initialize(self):
        """Initialize the agent"""
        self.logger.info(f"Initializing agent {self.agent_id}")
        
        # Register with message bus
        if self.message_bus:
            await self.message_bus.register_agent(self)
            await self.message_bus.subscribe(self.agent_id, self._handle_message)
            
        # Custom initialization
        await self._initialize()
        
        self.state = AgentState.READY
        self.logger.info(f"Agent {self.agent_id} initialized")
        
    @abstractmethod
    async def _initialize(self):
        """Custom initialization logic"""
        pass
        
    async def start(self):
        """Start the agent"""
        self.logger.info(f"Starting agent {self.agent_id}")
        
        # Start message processing
        asyncio.create_task(self._process_inbox())
        asyncio.create_task(self._process_outbox())
        
        # Start main loop
        asyncio.create_task(self._run())
        
        # Send heartbeat
        asyncio.create_task(self._heartbeat_loop())
        
    @abstractmethod
    async def _run(self):
        """Main agent loop"""
        pass
        
    async def _handle_message(self, message: Message):
        """Handle incoming message"""
        await self.inbox.put(message)
        self.metrics["messages_received"] += 1
        
    async def _process_inbox(self):
        """Process incoming messages"""
        while self.state not in [AgentState.SHUTTING_DOWN, AgentState.SHUTDOWN]:
            try:
                message = await asyncio.wait_for(self.inbox.get(), timeout=0.1)
                
                start_time = time.time()
                await self._process_message(message)
                processing_time = time.time() - start_time
                
                self.metrics["processing_times"].append(processing_time)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                self.metrics["errors"] += 1
                
    @abstractmethod
    async def _process_message(self, message: Message):
        """Process a single message"""
        pass
        
    async def _process_outbox(self):
        """Send outgoing messages"""
        while self.state not in [AgentState.SHUTTING_DOWN, AgentState.SHUTDOWN]:
            try:
                message = await asyncio.wait_for(self.outbox.get(), timeout=0.1)
                
                if self.message_bus:
                    await self.message_bus.publish(message)
                    
                self.metrics["messages_sent"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error sending message: {e}")
                
    async def send_message(self, recipient: Optional[str], 
                          message_type: MessageType,
                          payload: Dict[str, Any],
                          priority: int = 5,
                          requires_response: bool = False) -> Optional[str]:
        """Send a message to another agent"""
        message = Message(
            type=message_type,
            sender=self.agent_id,
            recipient=recipient,
            payload=payload,
            priority=priority,
            requires_ack=requires_response
        )
        
        if requires_response:
            self.pending_requests[message.id] = message
            
        await self.outbox.put(message)
        return message.id
        
    async def broadcast(self, message_type: MessageType, 
                       payload: Dict[str, Any],
                       priority: int = 5):
        """Broadcast a message to all agents"""
        await self.send_message(None, message_type, payload, priority)
        
    def update_state(self, new_state: AgentState, reason: str = ""):
        """Update agent state"""
        old_state = self.state
        self.state = new_state
        
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "old_state": old_state.value,
            "new_state": new_state.value,
            "reason": reason
        })
        
        self.logger.info(f"State change: {old_state.value} -> {new_state.value} ({reason})")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        avg_processing_time = 0
        if self.metrics["processing_times"]:
            avg_processing_time = sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"])
            
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "messages_sent": self.metrics["messages_sent"],
            "messages_received": self.metrics["messages_received"],
            "avg_processing_time": avg_processing_time,
            "errors": self.metrics["errors"]
        }
        
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.state not in [AgentState.SHUTTING_DOWN, AgentState.SHUTDOWN]:
            await self.broadcast(
                MessageType.HEARTBEAT,
                {"metrics": self.get_metrics()}
            )
            await asyncio.sleep(30)  # Every 30 seconds
            
    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        self.update_state(AgentState.SHUTTING_DOWN, "Shutdown requested")
        
        # Clean up resources
        await self._cleanup()
        
        # Unregister from message bus
        if self.message_bus:
            await self.message_bus.unregister_agent(self.agent_id)
            
        self.update_state(AgentState.SHUTDOWN, "Shutdown complete")
        
    @abstractmethod
    async def _cleanup(self):
        """Custom cleanup logic"""
        pass