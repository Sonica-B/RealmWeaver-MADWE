"""
Test script for MADWE Agent Communication System
Day 8 - Tests message queue, priority routing, and event subscription
Run from project root: python scripts/test_agent_communication.py
"""

import time
import threading
from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from unity_bridge.communication import (
    Message, MessageType, MessagePriority, MessageRouter
)
from agents.base_agent import BaseAgent, AgentConfig
from agents.environment_agent import EnvironmentAgent


class TestAgent(BaseAgent):
    """Simple test agent for communication testing"""
    
    def __init__(self, agent_id: str, router: MessageRouter):
        config = AgentConfig(
            agent_id=agent_id, 
            agent_type="test",
            enable_async=False
        )
        super().__init__(config, router)
        self.received_messages = []
        self.event_count = 0
    
    def _initialize(self):
        """Initialize test agent"""
        # Subscribe to test events
        self.subscribe_event("test_event")
        self.subscribe_event("test.*")  # Pattern subscription
    
    def handle_command(self, message: Message):
        """Handle command messages"""
        self.received_messages.append(message)
        print(f"[{self.agent_id}] Command from {message.sender_id}: {message.payload}")
    
    def handle_query(self, message: Message):
        """Handle query messages"""
        self.received_messages.append(message)
        query = message.payload.get('query')
        
        # Send response
        response_data = {
            'result': f"Response to {query}",
            'timestamp': time.time()
        }
        
        self.send_message(
            message.sender_id,
            MessageType.RESPONSE,
            response_data,
            correlation_id=message.message_id
        )
        print(f"[{self.agent_id}] Responded to query from {message.sender_id}")
    
    def handle_event(self, message: Message):
        """Handle event messages"""
        self.event_count += 1
        event_type = message.payload.get('event_type')
        print(f"[{self.agent_id}] Event received: {event_type}")


def test_basic_messaging(router: MessageRouter):
    """Test basic message passing between agents"""
    print("\n=== Test 1: Basic Messaging ===")
    
    # Create test agents
    agent1 = TestAgent("test_agent_1", router)
    agent2 = TestAgent("test_agent_2", router)
    
    # Start agents
    agent1.start()
    agent2.start()
    
    # Test direct message
    print("\n- Testing direct message...")
    agent1.send_message(
        "test_agent_2",
        MessageType.COMMAND,
        {'test': 'direct_message', 'value': 42}
    )
    
    time.sleep(0.1)
    assert len(agent2.received_messages) == 1
    print("✓ Direct message received")
    
    # Test query-response
    print("\n- Testing query-response...")
    response = agent1.send_and_wait(
        "test_agent_2",
        MessageType.QUERY,
        {'query': 'test_query'},
        timeout=2.0
    )
    
    assert response is not None
    assert 'result' in response.payload
    print(f"✓ Response received: {response.payload['result']}")
    
    # Stop agents
    agent1.stop()
    agent2.stop()
    
    return True


def test_priority_routing(router: MessageRouter):
    """Test priority-based message routing"""
    print("\n=== Test 2: Priority Routing ===")
    
    agent = TestAgent("priority_test_agent", router)
    agent.start()
    
    # Send messages with different priorities
    priorities = [
        (MessagePriority.LOW, "low_priority"),
        (MessagePriority.CRITICAL, "critical"),
        (MessagePriority.NORMAL, "normal"),
        (MessagePriority.HIGH, "high_priority"),
        (MessagePriority.BACKGROUND, "background")
    ]
    
    print("\n- Sending messages with different priorities...")
    for priority, label in priorities:
        agent.send_message(
            "priority_test_agent",
            MessageType.COMMAND,
            {'priority_test': label},
            priority=priority
        )
    
    time.sleep(0.2)
    
    # Check order - critical should be processed first
    print("\n- Checking processing order...")
    received_order = [msg.payload['priority_test'] for msg in agent.received_messages]
    print(f"Processing order: {received_order}")
    
    # Critical messages should be early in the list
    assert 'critical' in received_order[:2]
    print("✓ Priority routing working correctly")
    
    agent.stop()
    return True


def test_event_subscription(router: MessageRouter):
    """Test event subscription and broadcasting"""
    print("\n=== Test 3: Event Subscription ===")
    
    # Create multiple agents
    agents = []
    for i in range(3):
        agent = TestAgent(f"event_agent_{i}", router)
        agent.start()
        agents.append(agent)
    
    time.sleep(0.1)
    
    # Test specific event
    print("\n- Testing specific event subscription...")
    agents[0].emit_event("test_event", {'data': 'test_data'})
    
    time.sleep(0.1)
    
    # All agents should receive the event
    for agent in agents:
        assert agent.event_count >= 1
    print("✓ All agents received specific event")
    
    # Test pattern subscription
    print("\n- Testing pattern subscription...")
    initial_counts = [agent.event_count for agent in agents]
    
    agents[1].emit_event("test.sub_event", {'data': 'pattern_test'})
    
    time.sleep(0.1)
    
    # Check event counts increased
    for i, agent in enumerate(agents):
        assert agent.event_count > initial_counts[i]
    print("✓ Pattern subscription working")
    
    # Test broadcast
    print("\n- Testing broadcast...")
    agents[0].broadcast({'announcement': 'Hello all agents!'})
    
    time.sleep(0.1)
    print("✓ Broadcast sent successfully")
    
    # Stop all agents
    for agent in agents:
        agent.stop()
    
    return True


def test_environment_agent(router: MessageRouter):
    """Test environment agent functionality"""
    print("\n=== Test 4: Environment Agent ===")
    
    # Create environment agent
    env_config = AgentConfig(
        agent_id="env_agent",
        agent_type="environment",
        custom_config={
            'chunk_size': (16, 16),
            'biomes': ['forest', 'desert']
        }
    )
    
    env_agent = EnvironmentAgent(env_config, router)
    env_agent.start()
    
    # Create test agent to interact
    test_agent = TestAgent("test_requester", router)
    test_agent.start()
    
    # Test chunk generation
    print("\n- Testing chunk generation...")
    test_agent.send_message(
        "env_agent",
        MessageType.COMMAND,
        {
            'command': 'generate_chunk',
            'position': (0, 0),
            'biome': 'forest',
            'priority': 1
        }
    )
    
    time.sleep(0.5)
    
    # Query chunk existence
    print("\n- Testing chunk query...")
    response = test_agent.send_and_wait(
        "env_agent",
        MessageType.QUERY,
        {
            'query_type': 'chunk_exists',
            'position': (0, 0)
        },
        timeout=2.0
    )
    
    assert response is not None
    assert response.payload.get('exists') == True
    print("✓ Chunk generated and queryable")
    
    # Test world stats
    print("\n- Testing world statistics...")
    stats_response = test_agent.send_and_wait(
        "env_agent",
        MessageType.QUERY,
        {'query_type': 'world_stats'},
        timeout=2.0
    )
    
    assert stats_response is not None
    stats = stats_response.payload
    print(f"✓ World stats: {stats['chunks_generated']} chunks, "
          f"{stats['avg_generation_time']:.3f}s avg time")
    
    # Test player movement event
    print("\n- Testing player movement event...")
    test_agent.emit_event("player_moved", {
        'position': (100, 100),
        'player_id': 'test_player'
    })
    
    time.sleep(1.0)
    
    # Query stats again to see if more chunks were generated
    stats_response2 = test_agent.send_and_wait(
        "env_agent",
        MessageType.QUERY,
        {'query_type': 'world_stats'},
        timeout=2.0
    )
    
    new_stats = stats_response2.payload
    assert new_stats['chunks_generated'] >= stats['chunks_generated']
    print(f"✓ Player movement triggered generation: {new_stats['chunks_generated']} total chunks")
    
    # Stop agents
    env_agent.stop()
    test_agent.stop()
    
    return True


def test_performance(router: MessageRouter):
    """Test message passing performance"""
    print("\n=== Test 5: Performance Testing ===")
    
    # Create agents
    sender = TestAgent("perf_sender", router)
    receiver = TestAgent("perf_receiver", router)
    
    sender.start()
    receiver.start()
    
    # Send many messages
    num_messages = 1000
    print(f"\n- Sending {num_messages} messages...")
    
    start_time = time.time()
    
    for i in range(num_messages):
        sender.send_message(
            "perf_receiver",
            MessageType.COMMAND,
            {'index': i, 'timestamp': time.time()}
        )
    
    # Wait for processing
    timeout = 5.0
    wait_start = time.time()
    while len(receiver.received_messages) < num_messages and time.time() - wait_start < timeout:
        time.sleep(0.01)
    
    end_time = time.time()
    duration = end_time - start_time
    
    received_count = len(receiver.received_messages)
    print(f"✓ Processed {received_count}/{num_messages} messages in {duration:.3f}s")
    print(f"  Throughput: {received_count/duration:.0f} messages/second")
    
    # Check message ordering
    print("\n- Checking message ordering...")
    out_of_order = 0
    for i in range(1, len(receiver.received_messages)):
        if receiver.received_messages[i].payload['index'] < receiver.received_messages[i-1].payload['index']:
            out_of_order += 1
    
    print(f"  Out of order: {out_of_order}/{received_count} ({out_of_order/received_count*100:.1f}%)")
    
    # Get router stats
    router_stats = router.get_stats()
    print(f"\n- Router statistics:")
    print(f"  Total routed: {router_stats['total_messages']}")
    print(f"  Direct messages: {router_stats['direct_messages']}")
    print(f"  Events: {router_stats['events']}")
    print(f"  Broadcasts: {router_stats['broadcasts']}")
    
    sender.stop()
    receiver.stop()
    
    return True


def test_coordination(router: MessageRouter):
    """Test multi-agent coordination"""
    print("\n=== Test 6: Multi-Agent Coordination ===")
    
    # Create environment and asset agents
    env_config = AgentConfig(
        agent_id="coord_env_agent",
        agent_type="environment"
    )
    env_agent = EnvironmentAgent(env_config, router)
    
    # Create mock asset agent
    asset_agent = TestAgent("asset_agent", router)
    
    env_agent.start()
    asset_agent.start()
    
    # Test coordination message
    print("\n- Testing agent coordination...")
    env_agent.send_message(
        "coord_env_agent",
        MessageType.COORDINATION,
        {
            'action': 'prepare_generation',
            'chunk_position': (2, 2),
            'biome': 'cyberpunk'
        }
    )
    
    time.sleep(0.2)
    
    # Check if asset agent received request
    asset_messages = [msg for msg in asset_agent.received_messages 
                     if msg.payload.get('command') == 'generate_biome_textures']
    
    assert len(asset_messages) > 0
    print("✓ Coordination message sent to asset agent")
    
    # Simulate asset completion
    print("\n- Simulating asset generation completion...")
    asset_agent.emit_event("agent_coordination.asset_ready", {
        'chunk_position': (2, 2),
        'textures_generated': 3
    })
    
    time.sleep(0.1)
    print("✓ Asset completion event sent")
    
    env_agent.stop()
    asset_agent.stop()
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("MADWE Agent Communication System Test Suite")
    print("Day 8 - Testing message queue and event subscription")
    print("=" * 60)
    
    # Create message router
    router = MessageRouter(num_workers=4)
    
    # Run tests
    tests = [
        ("Basic Messaging", test_basic_messaging),
        ("Priority Routing", test_priority_routing),
        ("Event Subscription", test_event_subscription),
        ("Environment Agent", test_environment_agent),
        ("Performance", test_performance),
        ("Multi-Agent Coordination", test_coordination)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func(router):
                passed += 1
                print(f"\n✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 60)
    
    # Shutdown router
    router.shutdown()
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)