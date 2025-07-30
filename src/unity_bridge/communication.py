"""
Agent Communication System for MADWE
Day 8 - Production Code
"""

import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the system"""
    COMMAND = auto()
    QUERY = auto()
    RESPONSE = auto()
    EVENT = auto()
    BROADCAST = auto()
    STATE_UPDATE = auto()
    COORDINATION = auto()
    GENERATE_CONTENT = auto()
    VALIDATE_COHERENCE = auto()
    PREDICT_PLAYER = auto()


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class Message:
    """Message structure for agent communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: MessageType = MessageType.EVENT
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl: float = 60.0
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.priority.value < other.priority.value
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl
    
    def to_json(self) -> str:
        return json.dumps({
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.name,
            "priority": self.priority.name,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        data = json.loads(json_str)
        return cls(
            message_id=data.get("message_id"),
            sender_id=data.get("sender_id", ""),
            recipient_id=data.get("recipient_id", ""),
            message_type=MessageType[data.get("message_type", "EVENT")],
            priority=MessagePriority[data.get("priority", "NORMAL")],
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            ttl=data.get("ttl", 60.0),
            correlation_id=data.get("correlation_id"),
            metadata=data.get("metadata", {})
        )


class MessageQueue:
    """Priority-based message queue with expiration handling"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._message_count = 0
        self._dropped_count = 0
        
    def put(self, message: Message) -> bool:
        with self._lock:
            if len(self._queue) >= self.max_size:
                if self._queue and message.priority.value < self._queue[-1][0]:
                    heapq.heappop(self._queue)
                    self._dropped_count += 1
                else:
                    self._dropped_count += 1
                    return False
            
            heapq.heappush(self._queue, (message.priority.value, time.time(), message))
            self._message_count += 1
            self._not_empty.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Message]:
        with self._lock:
            end_time = time.time() + timeout if timeout else None
            
            while True:
                self._clean_expired()
                
                if self._queue:
                    _, _, message = heapq.heappop(self._queue)
                    return message
                
                if timeout is None:
                    return None
                
                remaining = end_time - time.time() if end_time else None
                if remaining is not None and remaining <= 0:
                    return None
                
                self._not_empty.wait(remaining)
    
    def _clean_expired(self):
        """Remove expired messages"""
        current_time = time.time()
        cleaned = []
        
        while self._queue:
            priority, timestamp, message = heapq.heappop(self._queue)
            if not message.is_expired():
                cleaned.append((priority, timestamp, message))
        
        for item in cleaned:
            heapq.heappush(self._queue, item)
    
    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                'queue_size': len(self._queue),
                'total_messages': self._message_count,
                'dropped_messages': self._dropped_count
            }


class EventSubscription:
    """Manages event subscriptions with pattern matching"""
    
    def __init__(self):
        self._subscriptions: Dict[str, List[Tuple[str, Callable]]] = defaultdict(list)
        self._pattern_subscriptions: List[Tuple[str, Callable, str]] = []
        self._lock = threading.RLock()
    
    def subscribe(self, event_type: str, agent_id: str, callback: Callable):
        with self._lock:
            if '*' in event_type:
                self._pattern_subscriptions.append((event_type, callback, agent_id))
            else:
                self._subscriptions[event_type].append((agent_id, callback))
    
    def get_subscribers(self, event_type: str) -> List[Callable]:
        with self._lock:
            subscribers = []
            
            # Direct subscriptions
            for agent_id, callback in self._subscriptions.get(event_type, []):
                subscribers.append(callback)
            
            # Pattern subscriptions
            for pattern, callback, _ in self._pattern_subscriptions:
                if self._match_pattern(pattern, event_type):
                    subscribers.append(callback)
            
            return subscribers
    
    def _match_pattern(self, pattern: str, event_type: str) -> bool:
        if '*' not in pattern:
            return pattern == event_type
        
        parts = pattern.split('*')
        if not event_type.startswith(parts[0]):
            return False
        if len(parts) > 1 and not event_type.endswith(parts[-1]):
            return False
        return True


class MessageRouter:
    """Routes messages between agents with priority handling"""
    
    def __init__(self, num_workers: int = 4):
        self._agent_queues: Dict[str, MessageQueue] = {}
        self._broadcast_queue = MessageQueue(max_size=1000)
        self._event_subscriptions = EventSubscription()
        self._routing_table: Dict[str, str] = {}
        self._workers = ThreadPoolExecutor(max_workers=num_workers)
        self._running = False
        self._stats = defaultdict(int)
        self._lock = threading.RLock()
        
    def register_agent(self, agent_id: str, queue_size: int = 1000):
        with self._lock:
            if agent_id not in self._agent_queues:
                self._agent_queues[agent_id] = MessageQueue(max_size=queue_size)
                self._routing_table[agent_id] = agent_id
                logger.info(f"Registered agent: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        with self._lock:
            self._agent_queues.pop(agent_id, None)
            self._routing_table.pop(agent_id, None)
            logger.info(f"Unregistered agent: {agent_id}")
    
    def route_message(self, message: Message) -> bool:
        with self._lock:
            self._stats['total_messages'] += 1
            
            # Handle broadcasts
            if message.message_type == MessageType.BROADCAST or not message.recipient_id:
                self._stats['broadcasts'] += 1
                return self._broadcast_queue.put(message)
            
            # Handle events
            if message.message_type == MessageType.EVENT:
                self._stats['events'] += 1
                self._handle_event(message)
                return True
            
            # Direct routing
            if message.recipient_id in self._agent_queues:
                self._stats['direct_messages'] += 1
                return self._agent_queues[message.recipient_id].put(message)
            
            self._stats['unroutable'] += 1
            logger.warning(f"Cannot route message to unknown agent: {message.recipient_id}")
            return False
    
    def _handle_event(self, message: Message):
        event_type = message.payload.get('event_type', 'unknown')
        subscribers = self._event_subscriptions.get_subscribers(event_type)
        
        for callback in subscribers:
            self._workers.submit(self._safe_callback, callback, message)
    
    def _safe_callback(self, callback: Callable, message: Message):
        try:
            callback(message)
        except Exception as e:
            logger.error(f"Error in event callback: {e}")
    
    def get_message(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        with self._lock:
            if agent_id not in self._agent_queues:
                return None
        
        # Try direct queue first
        message = self._agent_queues[agent_id].get(timeout=timeout)
        if message:
            return message
        
        # If no direct message, check broadcast queue
        return self._broadcast_queue.get(timeout=0)
    
    def subscribe_event(self, event_type: str, agent_id: str, callback: Callable):
        self._event_subscriptions.subscribe(event_type, agent_id, callback)
        self._stats['subscriptions'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
            stats['agent_count'] = len(self._agent_queues)
            stats['queue_stats'] = {}
            
            for agent_id, queue in self._agent_queues.items():
                stats['queue_stats'][agent_id] = queue.get_stats()
            
            return stats
    
    def shutdown(self):
        self._running = False
        self._workers.shutdown(wait=True)
        logger.info("Message router shutdown complete")