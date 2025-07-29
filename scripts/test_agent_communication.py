"""
Simple test script for MADWE Agent Communication
Run from project root: python scripts/test_communication.py
"""

import sys
import os
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ..src.unity_bridge.communication import (
    Message, MessageType, MessagePriority, MessageRouter
)
from ..src.agents.base_agent import BaseAgent, AgentConfig


class TestAgent(BaseAgent):
    """Simple test agent"""
    
    def __init__(self, agent_id: str, router: MessageRouter):
        config = AgentConfig(agent_id=agent_id, agent_type="test")
        super().__init__(config, router)
        self.received = 0
    
    def _initialize(self):
        pass
    
    def handle_command(self, message: Message):
        self.received += 1
        print(f"{self.agent_id} received command from {message.sender_id}")
    
    def handle_query(self, message: Message):
        self.received += 1
        # Send response
        self.send_message(
            message.sender_id,
            MessageType.RESPONSE,
            {'result': 'query_response'},
            correlation_id=message.message_id
        )
    
    def handle_state_update(self, message: Message):
        self.received += 1


def main():
    print("Testing MADWE Agent Communication System")
    print("=" * 50)
    
    # Create router
    router = MessageRouter(num_workers=4)
    
    # Create agents
    agent1 = TestAgent("agent_1", router)
    agent2 = TestAgent("agent_2", router)
    
    # Start agents
    agent1.start()
    agent2.start()
    
    print("\n1. Testing direct messaging...")
    # Test direct message
    agent1.send_message(
        "agent_2",
        MessageType.COMMAND,
        {'test': 'direct_message'}
    )
    
    time.sleep(0.5)
    
    print("\n2. Testing request-response...")
    # Test request-response
    response = agent1.send_and_wait(
        "agent_2",
        MessageType.QUERY,
        {'query': 'test_query'},
        timeout=1.0
    )
    
    if response:
        print(f"Got response: {response.payload}")
    
    print("\n3. Testing broadcast...")
    # Test broadcast
    agent1.broadcast({'announcement': 'hello_all'})
    
    time.sleep(0.5)
    
    print("\n4. Testing events...")
    # Test events
    def on_test_event(msg):
        print(f"Event received: {msg.payload}")
    
    agent2.subscribe_event("test_event", on_test_event)
    agent1.publish_event("test_event", {'data': 'event_data'})
    
    time.sleep(0.5)
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Statistics:")
    print(f"Agent 1 received: {agent1.received} messages")
    print(f"Agent 2 received: {agent2.received} messages")
    
    router_stats = router.get_stats()
    print(f"\nRouter stats:")
    print(f"  Total messages: {router_stats['total_messages']}")
    print(f"  Direct messages: {router_stats['direct_messages']}")
    print(f"  Broadcasts: {router_stats['broadcasts']}")
    print(f"  Events: {router_stats['events']}")
    
    # Stop agents
    agent1.stop()
    agent2.stop()
    router.shutdown()
    
    print("\nTest complete!")


if __name__ == "__main__":
    main()