"""
Base Agent Framework for MADWE
Day 5: Friday, June 7 - Multi-Agent Foundation
Provides the foundation for all agents in the multi-agent system
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime
import uuid
from pathlib import Path
import pickle
from collections import deque
import threading
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
    correlation_id: Optional[str] = None  # For request-response tracking
    priority: int = 5  # 1-10, 1 being highest priority
    ttl: Optional[int] = None  # Time to live in seconds
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
            "requires_ack": self.requires_ack,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary"""
        data["type"] = MessageType(data["type"])
        return cls(**data)


T = TypeVar("T")


class StateManager(Generic[T]):
    """Generic state management for agents"""

    def __init__(self, initial_state: T):
        self._state: T = initial_state
        self._lock = threading.RLock()
        self._observers: List[Callable[[T], None]] = []
        self._history: deque = deque(maxlen=100)
        self._checkpoints: Dict[str, T] = {}

    @property
    def state(self) -> T:
        """Get current state"""
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: T):
        """Set state and notify observers"""
        with self._lock:
            old_state = self._state
            self._state = value
            self._history.append((datetime.now(), old_state, value))

            # Notify observers
            for observer in self._observers:
                try:
                    observer(value)
                except Exception as e:
                    logging.error(f"Observer error: {e}")

    def update(self, updater: Callable[[T], T]):
        """Update state using a function"""
        with self._lock:
            self.state = updater(self._state)

    def observe(self, observer: Callable[[T], None]):
        """Add state observer"""
        self._observers.append(observer)

    def checkpoint(self, name: str):
        """Create a named checkpoint"""
        with self._lock:
            self._checkpoints[name] = pickle.loads(pickle.dumps(self._state))

    def restore(self, name: str):
        """Restore from checkpoint"""
        with self._lock:
            if name in self._checkpoints:
                self.state = self._checkpoints[name]
            else:
                raise ValueError(f"Checkpoint '{name}' not found")

    def get_history(self, limit: int = 10) -> List[Tuple[str, Any, Any]]:
        """Get state history"""
        with self._lock:
            history = list(self._history)[-limit:]
            return [(t.isoformat(), old, new) for t, old, new in history]


class MessageBus:
    """Central message bus for agent communication"""

    def __init__(self):
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._topic_subscribers: Dict[str, List[str]] = {}
        self._message_log: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._metrics = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_dropped": 0,
        }

    async def register(self, agent_id: str) -> asyncio.Queue:
        """Register an agent and return its message queue"""
        with self._lock:
            if agent_id not in self._subscribers:
                self._subscribers[agent_id] = asyncio.Queue(maxsize=1000)
            return self._subscribers[agent_id]

    async def unregister(self, agent_id: str):
        """Unregister an agent"""
        with self._lock:
            if agent_id in self._subscribers:
                del self._subscribers[agent_id]

            # Remove from topic subscriptions
            for topic, subscribers in self._topic_subscribers.items():
                if agent_id in subscribers:
                    subscribers.remove(agent_id)

    async def publish(self, message: Message):
        """Publish a message"""
        with self._lock:
            self._message_log.append((datetime.now(), message))
            self._metrics["messages_sent"] += 1

            if message.recipient:
                # Direct message
                if message.recipient in self._subscribers:
                    try:
                        await self._subscribers[message.recipient].put(message)
                        self._metrics["messages_delivered"] += 1
                    except asyncio.QueueFull:
                        self._metrics["messages_dropped"] += 1
                        logging.warning(
                            f"Queue full for {message.recipient}, message dropped"
                        )
            else:
                # Broadcast message
                delivered = 0
                for agent_id, queue in self._subscribers.items():
                    if agent_id != message.sender:  # Don't send to self
                        try:
                            await queue.put(message)
                            delivered += 1
                        except asyncio.QueueFull:
                            self._metrics["messages_dropped"] += 1

                self._metrics["messages_delivered"] += delivered

    def subscribe_to_topic(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic"""
        with self._lock:
            if topic not in self._topic_subscribers:
                self._topic_subscribers[topic] = []
            if agent_id not in self._topic_subscribers[topic]:
                self._topic_subscribers[topic].append(agent_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get message bus metrics"""
        with self._lock:
            return self._metrics.copy()


class BaseAgent(ABC):
    """Base class for all agents in the MADWE system"""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
        message_bus: Optional[MessageBus] = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}

        # Setup logging
        self.logger = logging.getLogger(f"{agent_type}.{agent_id}")

        # State management
        self.state = AgentState.INITIALIZING
        self._state_manager = StateManager({"status": "initializing"})

        # Message handling
        self.message_bus = message_bus or MessageBus()
        self.message_queue: Optional[asyncio.Queue] = None
        self.message_handlers: Dict[MessageType, Callable] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}

        # Performance monitoring
        self.metrics = {
            "messages_processed": 0,
            "messages_sent": 0,
            "errors": 0,
            "uptime": 0,
            "last_activity": None,
        }

        # Lifecycle management
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._start_time = None

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        self.register_handler(MessageType.STATUS_QUERY, self._handle_status_query)
        self.register_handler(MessageType.COMMAND, self._handle_command)

    async def _handle_heartbeat(self, message: Message):
        """Handle heartbeat messages"""
        # Respond with heartbeat acknowledgment
        await self.send_message(
            Message(
                type=MessageType.HEARTBEAT,
                recipient=message.sender,
                correlation_id=message.id,
                payload={"status": self.state.value, "uptime": self.get_uptime()},
            )
        )

    async def _handle_status_query(self, message: Message):
        """Handle status query messages"""
        status = await self.get_status()
        await self.send_message(
            Message(
                type=MessageType.STATUS_RESPONSE,
                recipient=message.sender,
                correlation_id=message.id,
                payload=status,
            )
        )

    async def _handle_command(self, message: Message):
        """Handle command messages"""
        command = message.payload.get("command")

        if command == "pause":
            await self.pause()
        elif command == "resume":
            await self.resume()
        elif command == "shutdown":
            await self.shutdown()
        else:
            self.logger.warning(f"Unknown command: {command}")

    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler

    async def initialize(self):
        """Initialize the agent"""
        self.logger.info(f"Initializing agent {self.agent_id}")

        # Register with message bus
        self.message_queue = await self.message_bus.register(self.agent_id)

        # Perform agent-specific initialization
        await self._initialize()

        self.state = AgentState.READY
        self._start_time = datetime.now()
        self.logger.info(f"Agent {self.agent_id} initialized")

    @abstractmethod
    async def _initialize(self):
        """Agent-specific initialization"""
        pass

    async def start(self):
        """Start the agent"""
        if self._running:
            return

        self.logger.info(f"Starting agent {self.agent_id}")
        self._running = True

        # Start message processing
        message_task = asyncio.create_task(self._process_messages())
        self._tasks.append(message_task)

        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._tasks.append(heartbeat_task)

        # Start agent-specific tasks
        await self._start()

        self.logger.info(f"Agent {self.agent_id} started")

    @abstractmethod
    async def _start(self):
        """Agent-specific start logic"""
        pass

    async def _process_messages(self):
        """Process incoming messages"""
        while self._running:
            try:
                # Get message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)

                self.metrics["messages_processed"] += 1
                self.metrics["last_activity"] = datetime.now()

                # Check TTL
                if message.ttl:
                    age = (
                        datetime.now() - datetime.fromisoformat(message.timestamp)
                    ).total_seconds()
                    if age > message.ttl:
                        self.logger.debug(
                            f"Message {message.id} expired (TTL: {message.ttl}s)"
                        )
                        continue

                # Process message
                await self._handle_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                self.metrics["errors"] += 1

    async def _handle_message(self, message: Message):
        """Handle a single message"""
        # Check if this is a response to a pending request
        if message.correlation_id in self._pending_responses:
            future = self._pending_responses[message.correlation_id]
            future.set_result(message)
            del self._pending_responses[message.correlation_id]
            return

        # Handle based on message type
        handler = self.message_handlers.get(message.type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")

                # Send error response if required
                if message.requires_ack:
                    await self.send_message(
                        Message(
                            type=MessageType.ERROR,
                            recipient=message.sender,
                            correlation_id=message.id,
                            payload={"error": str(e)},
                        )
                    )
        else:
            self.logger.warning(f"No handler for message type: {message.type}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self._running:
            await asyncio.sleep(30)  # Every 30 seconds

            # Broadcast heartbeat
            await self.send_message(
                Message(
                    type=MessageType.HEARTBEAT,
                    payload={
                        "agent_id": self.agent_id,
                        "agent_type": self.agent_type,
                        "status": self.state.value,
                        "uptime": self.get_uptime(),
                    },
                )
            )

    async def send_message(self, message: Message) -> Optional[Message]:
        """Send a message"""
        message.sender = self.agent_id
        self.metrics["messages_sent"] += 1

        # If expecting response, create future
        if message.requires_ack:
            future = asyncio.Future()
            self._pending_responses[message.id] = future

        await self.message_bus.publish(message)

        # Wait for response if required
        if message.requires_ack:
            try:
                response = await asyncio.wait_for(future, timeout=10.0)
                return response
            except asyncio.TimeoutError:
                del self._pending_responses[message.id]
                self.logger.warning(f"Timeout waiting for response to {message.id}")
                return None

        return None

    async def request(
        self, recipient: str, payload: Dict[str, Any], timeout: float = 10.0
    ) -> Optional[Dict[str, Any]]:
        """Send a request and wait for response"""
        message = Message(
            type=MessageType.REQUEST,
            recipient=recipient,
            payload=payload,
            requires_ack=True,
        )

        response = await self.send_message(message)
        if response:
            return response.payload
        return None

    async def broadcast(self, payload: Dict[str, Any]):
        """Broadcast a message to all agents"""
        await self.send_message(Message(type=MessageType.BROADCAST, payload=payload))

    async def pause(self):
        """Pause the agent"""
        self.logger.info(f"Pausing agent {self.agent_id}")
        self.state = AgentState.PAUSED
        await self._pause()

    @abstractmethod
    async def _pause(self):
        """Agent-specific pause logic"""
        pass

    async def resume(self):
        """Resume the agent"""
        self.logger.info(f"Resuming agent {self.agent_id}")
        self.state = AgentState.READY
        await self._resume()

    @abstractmethod
    async def _resume(self):
        """Agent-specific resume logic"""
        pass

    async def shutdown(self):
        """Shutdown the agent"""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        self.state = AgentState.SHUTTING_DOWN
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Agent-specific shutdown
        await self._shutdown()

        # Unregister from message bus
        await self.message_bus.unregister(self.agent_id)

        self.state = AgentState.SHUTDOWN
        self.logger.info(f"Agent {self.agent_id} shutdown complete")

    @abstractmethod
    async def _shutdown(self):
        """Agent-specific shutdown logic"""
        pass

    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "uptime": self.get_uptime(),
            "metrics": self.metrics.copy(),
            "custom_status": await self._get_custom_status(),
        }

    @abstractmethod
    async def _get_custom_status(self) -> Dict[str, Any]:
        """Get agent-specific status"""
        pass

    def get_uptime(self) -> float:
        """Get agent uptime in seconds"""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return 0

    def save_state(self, filepath: Path):
        """Save agent state to file"""
        state_data = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state.value,
            "metrics": self.metrics,
            "custom_state": self._get_custom_state(),
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2)

    @abstractmethod
    def _get_custom_state(self) -> Dict[str, Any]:
        """Get agent-specific state for saving"""
        pass

    def load_state(self, filepath: Path):
        """Load agent state from file"""
        with open(filepath, "r") as f:
            state_data = json.load(f)

        self.metrics = state_data.get("metrics", {})
        self._load_custom_state(state_data.get("custom_state", {}))

    @abstractmethod
    def _load_custom_state(self, state: Dict[str, Any]):
        """Load agent-specific state"""
        pass


class AgentCluster:
    """Manages a cluster of agents"""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.message_bus = MessageBus()
        self.logger = logging.getLogger("AgentCluster")

    def add_agent(self, agent: BaseAgent):
        """Add an agent to the cluster"""
        agent.message_bus = self.message_bus
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Added agent {agent.agent_id} to cluster")

    async def start_all(self):
        """Start all agents"""
        self.logger.info("Starting all agents")

        # Initialize all agents
        init_tasks = [agent.initialize() for agent in self.agents.values()]
        await asyncio.gather(*init_tasks)

        # Start all agents
        start_tasks = [agent.start() for agent in self.agents.values()]
        await asyncio.gather(*start_tasks)

        self.logger.info(f"Started {len(self.agents)} agents")

    async def shutdown_all(self):
        """Shutdown all agents"""
        self.logger.info("Shutting down all agents")

        shutdown_tasks = [agent.shutdown() for agent in self.agents.values()]
        await asyncio.gather(*shutdown_tasks)

        self.logger.info("All agents shutdown")

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status_tasks = [agent.get_status() for agent in self.agents.values()]
        statuses = await asyncio.gather(*status_tasks)

        return {
            "agents": statuses,
            "message_bus_metrics": self.message_bus.get_metrics(),
        }

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)


# Example implementation of a simple agent
class EchoAgent(BaseAgent):
    """Simple agent that echoes messages"""

    async def _initialize(self):
        """Initialize echo agent"""
        self.register_handler(MessageType.REQUEST, self._handle_echo_request)
        self.echo_count = 0

    async def _start(self):
        """Start echo agent"""
        pass

    async def _pause(self):
        """Pause echo agent"""
        pass

    async def _resume(self):
        """Resume echo agent"""
        pass

    async def _shutdown(self):
        """Shutdown echo agent"""
        pass

    async def _handle_echo_request(self, message: Message):
        """Handle echo request"""
        self.echo_count += 1

        # Echo the payload back
        await self.send_message(
            Message(
                type=MessageType.RESPONSE,
                recipient=message.sender,
                correlation_id=message.id,
                payload={"echo": message.payload, "echo_count": self.echo_count},
            )
        )

    async def _get_custom_status(self) -> Dict[str, Any]:
        """Get echo agent status"""
        return {"echo_count": self.echo_count}

    def _get_custom_state(self) -> Dict[str, Any]:
        """Get echo agent state"""
        return {"echo_count": self.echo_count}

    def _load_custom_state(self, state: Dict[str, Any]):
        """Load echo agent state"""
        self.echo_count = state.get("echo_count", 0)


async def test_agent_system():
    """Test the agent system"""
    # Create cluster
    cluster = AgentCluster()

    # Create agents
    echo1 = EchoAgent("echo1", "echo")
    echo2 = EchoAgent("echo2", "echo")

    # Add to cluster
    cluster.add_agent(echo1)
    cluster.add_agent(echo2)

    # Start cluster
    await cluster.start_all()

    # Send test message
    response = await echo1.request("echo2", {"message": "Hello from echo1!"})
    print(f"Response: {response}")

    # Get cluster status
    status = await cluster.get_cluster_status()
    print(f"Cluster status: {json.dumps(status, indent=2)}")

    # Shutdown
    await cluster.shutdown_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_agent_system())
