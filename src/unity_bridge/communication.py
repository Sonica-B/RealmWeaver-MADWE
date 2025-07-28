"""
Unity-Python Bridge Communication System
Day 2: Real-time communication between Unity and Python
"""

import socket
import threading
import json
import queue
import time
import struct
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
import logging
import numpy as np
import asyncio


@dataclass
class UnityMessage:
    """Message structure for Unity communication"""
    msg_type: str
    data: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
            
    def to_bytes(self) -> bytes:
        """Convert to bytes for transmission"""
        json_str = json.dumps({
            'type': self.msg_type,
            'data': self.data,
            'timestamp': self.timestamp
        })
        
        # Prefix with length for proper framing
        json_bytes = json_str.encode('utf-8')
        length_prefix = struct.pack('!I', len(json_bytes))
        return length_prefix + json_bytes
        
    @classmethod
    def from_bytes(cls, data: bytes) -> 'UnityMessage':
        """Create message from bytes"""
        json_str = data.decode('utf-8')
        msg_dict = json.loads(json_str)
        return cls(
            msg_type=msg_dict['type'],
            data=msg_dict['data'],
            timestamp=msg_dict.get('timestamp', time.time())
        )


class UnityBridge:
    """Main Unity-Python communication bridge"""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 5005):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.connected = False
        
        # Message queues
        self.send_queue = queue.Queue(maxsize=1000)
        self.receive_queue = queue.Queue(maxsize=1000)
        
        # Message handlers
        self.handlers: Dict[str, Callable] = {}
        
        # Threading
        self.running = False
        self.server_thread = None
        self.receive_thread = None
        self.send_thread = None
        
        # Performance tracking
        self.latency_buffer = []
        self.max_latency_samples = 100
        
        # Logging
        self.logger = logging.getLogger('UnityBridge')
        
    def start(self):
        """Start the bridge server"""
        self.running = True
        
        # Create server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        
        # Start server thread
        self.server_thread = threading.Thread(target=self._server_loop)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.logger.info(f"Unity bridge started on {self.host}:{self.port}")
        
    def stop(self):
        """Stop the bridge"""
        self.running = False
        
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
            
        # Wait for threads to finish
        if self.server_thread:
            self.server_thread.join(timeout=1)
        if self.receive_thread:
            self.receive_thread.join(timeout=1)
        if self.send_thread:
            self.send_thread.join(timeout=1)
            
        self.logger.info("Unity bridge stopped")
        
    def _server_loop(self):
        """Main server loop accepting connections"""
        while self.running:
            try:
                # Set timeout to allow periodic checks
                self.server_socket.settimeout(1.0)
                
                try:
                    client_socket, address = self.server_socket.accept()
                    self.logger.info(f"Unity connected from {address}")
                    
                    # Handle new connection
                    self._handle_connection(client_socket)
                    
                except socket.timeout:
                    continue
                    
            except Exception as e:
                if self.running:
                    self.logger.error(f"Server error: {e}")
                    
    def _handle_connection(self, client_socket: socket.socket):
        """Handle Unity client connection"""
        self.client_socket = client_socket
        self.client_socket.settimeout(0.1)  # Non-blocking with timeout
        self.connected = True
        
        # Start communication threads
        self.receive_thread = threading.Thread(target=self._receive_loop)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        self.send_thread = threading.Thread(target=self._send_loop)
        self.send_thread.daemon = True
        self.send_thread.start()
        
        # Send initial handshake
        handshake = UnityMessage(
            msg_type='handshake',
            data={'status': 'ready', 'version': '1.0'}
        )
        self.send_message(handshake)
        
    def _receive_loop(self):
        """Receive messages from Unity"""
        buffer = b''
        
        while self.running and self.connected:
            try:
                # Receive data
                data = self.client_socket.recv(4096)
                
                if not data:
                    # Connection closed
                    self.connected = False
                    break
                    
                buffer += data
                
                # Process complete messages
                while len(buffer) >= 4:
                    # Read message length
                    msg_length = struct.unpack('!I', buffer[:4])[0]
                    
                    if len(buffer) >= 4 + msg_length:
                        # Extract complete message
                        msg_data = buffer[4:4 + msg_length]
                        buffer = buffer[4 + msg_length:]
                        
                        # Parse message
                        try:
                            message = UnityMessage.from_bytes(msg_data)
                            
                            # Track latency
                            if 'sent_time' in message.data:
                                latency = (time.time() - message.data['sent_time']) * 1000
                                self._track_latency(latency)
                                
                            # Add to receive queue
                            self.receive_queue.put(message)
                            
                            # Process handlers
                            self._process_message(message)
                            
                        except Exception as e:
                            self.logger.error(f"Message parse error: {e}")
                    else:
                        # Wait for more data
                        break
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"Receive error: {e}")
                    self.connected = False
                    break
                    
    def _send_loop(self):
        """Send messages to Unity"""
        while self.running and self.connected:
            try:
                # Get message from queue with timeout
                message = self.send_queue.get(timeout=0.1)
                
                # Add timing info
                message.data['sent_time'] = time.time()
                
                # Send message
                msg_bytes = message.to_bytes()
                self.client_socket.sendall(msg_bytes)
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    self.logger.error(f"Send error: {e}")
                    self.connected = False
                    break
                    
    def _process_message(self, message: UnityMessage):
        """Process received message with handlers"""
        handler = self.handlers.get(message.msg_type)
        if handler:
            try:
                handler(message)
            except Exception as e:
                self.logger.error(f"Handler error for {message.msg_type}: {e}")
                
    def send_message(self, message: UnityMessage):
        """Send message to Unity"""
        if not self.connected:
            self.logger.warning("Not connected to Unity")
            return
            
        try:
            self.send_queue.put(message, timeout=0.1)
        except queue.Full:
            self.logger.warning("Send queue full, dropping message")
            
    def register_handler(self, msg_type: str, handler: Callable[[UnityMessage], None]):
        """Register message handler"""
        self.handlers[msg_type] = handler
        
    def _track_latency(self, latency_ms: float):
        """Track communication latency"""
        self.latency_buffer.append(latency_ms)
        if len(self.latency_buffer) > self.max_latency_samples:
            self.latency_buffer.pop(0)
            
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latency_buffer:
            return {'avg': 0, 'min': 0, 'max': 0}
            
        return {
            'avg': np.mean(self.latency_buffer),
            'min': np.min(self.latency_buffer),
            'max': np.max(self.latency_buffer),
            'p95': np.percentile(self.latency_buffer, 95)
        }
        
    def send_tile_update(self, tiles: np.ndarray, position: tuple[int, int]):
        """Send tile update to Unity"""
        message = UnityMessage(
            msg_type='tile_update',
            data={
                'tiles': tiles.tolist(),
                'position': list(position),
                'size': list(tiles.shape),
                'biome': 'forest'  # You'll need to pass this as parameter
            }
        )
        self.send_message(message)
        
    def send_asset_generated(self, asset_id: str, asset_path: str, metadata: Dict[str, Any]):
        """Notify Unity of generated asset"""
        message = UnityMessage(
            msg_type='asset_generated',
            data={
                'asset_id': asset_id,
                'path': asset_path,
                'metadata': metadata
            }
        )
        self.send_message(message)
        
    def request_player_state(self) -> Optional[Dict[str, Any]]:
        """Request current player state from Unity"""
        request = UnityMessage(
            msg_type='get_player_state',
            data={'request_id': str(time.time())}
        )
        
        # Send request
        self.send_message(request)
        
        # Wait for response (simplified - in production use correlation IDs)
        start_time = time.time()
        timeout = 0.1  # 100ms timeout
        
        while time.time() - start_time < timeout:
            try:
                msg = self.receive_queue.get(timeout=0.01)
                if msg.msg_type == 'player_state':
                    return msg.data
            except queue.Empty:
                continue
                
        return None


class UnityBridgeAsync:
    """Async version of Unity bridge for integration with async agents"""
    
    def __init__(self, bridge: UnityBridge):
        self.bridge = bridge
        self.response_futures: Dict[str, asyncio.Future] = {}
        
    async def send_and_wait(self, message: UnityMessage, timeout: float = 0.1) -> Optional[UnityMessage]:
        """Send message and wait for response"""
        request_id = str(time.time())
        message.data['request_id'] = request_id
        
        # Create future for response
        future = asyncio.Future()
        self.response_futures[request_id] = future
        
        # Send message
        self.bridge.send_message(message)
        
        try:
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            return None
        finally:
            # Clean up
            self.response_futures.pop(request_id, None)
            
    def handle_response(self, message: UnityMessage):
        """Handle response messages"""
        request_id = message.data.get('request_id')
        if request_id and request_id in self.response_futures:
            self.response_futures[request_id].set_result(message)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create and start bridge
    bridge = UnityBridge()
    bridge.start()
    
    # Register handlers
    def handle_player_action(message: UnityMessage):
        print(f"Player action: {message.data}")
        
    bridge.register_handler('player_action', handle_player_action)
    
    # Keep running
    try:
        while True:
            time.sleep(1)
            
            # Print stats
            if bridge.connected:
                stats = bridge.get_latency_stats()
                print(f"Latency: {stats['avg']:.1f}ms (min: {stats['min']:.1f}, max: {stats['max']:.1f})")
                
    except KeyboardInterrupt:
        bridge.stop()