"""
Simple test to verify agent messaging works
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import time
from unity_bridge.communication import Message, MessageType, MessagePriority, MessageRouter

# Create router
router = MessageRouter()

# Register two agents
router.register_agent("agent1", queue_size=100)
router.register_agent("agent2", queue_size=100)

print("1. Testing direct message...")
# Create and route a message
msg = Message(
    sender_id="agent1",
    recipient_id="agent2",
    message_type=MessageType.COMMAND,
    payload={"test": "hello"}
)
success = router.route_message(msg)
print(f"   Route message: {success}")

# Get message from agent2's queue
received = router.get_message("agent2", timeout=1.0)
print(f"   Message received: {received is not None}")
if received:
    print(f"   Payload: {received.payload}")

print("\n2. Testing event...")
# Create event message
event_msg = Message(
    sender_id="agent1",
    message_type=MessageType.EVENT,
    payload={"event_type": "test_event", "data": "event data"}
)

# Subscribe to event
def event_handler(msg):
    print(f"   Event handled: {msg.payload}")

router.subscribe_event("test_event", "agent2", event_handler)

# Route event
router.route_message(event_msg)
time.sleep(0.1)  # Give time for async handling

print("\n3. Testing priority...")
# Send messages with different priorities
for priority, label in [(MessagePriority.LOW, "low"), 
                       (MessagePriority.CRITICAL, "critical"),
                       (MessagePriority.NORMAL, "normal")]:
    msg = Message(
        sender_id="agent1",
        recipient_id="agent2",
        message_type=MessageType.COMMAND,
        priority=priority,
        payload={"priority": label}
    )
    router.route_message(msg)

# Get messages and check order
print("   Message order:")
for i in range(3):
    msg = router.get_message("agent2", timeout=0.1)
    if msg:
        print(f"   {i+1}. {msg.payload['priority']} (priority: {msg.priority.name})")

print("\nAll basic tests completed!")
router.shutdown()