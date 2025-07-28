# src/unity_integration/unity_bridge.py
import asyncio
import json
import time
from typing import Dict, Any, Callable
import threading
from queue import Queue, Empty
import struct
import socket

class UnityBridge:
    """
    High-performance Unity-Python bridge using named pipes
    Target: <50ms latency for real-time communication
    """
    
    def __init__(self, pipe_name: str = "madwe_pipe", port: int = 12345):
        self.pipe_name = pipe_name
        self.port = port
        self.socket = None
        self.is_connected = False
        
        # Message queues
        self.outbound_queue = Queue()
        self.inbound_queue = Queue()
        
        # Callbacks
        self.message_handlers = {}
        
        # Performance monitoring
        self.message_times = []
        self.request_cache = {}  # Caching layer
        
        # Threading
        self._running = False
        self._threads = []
    
    def start(self):
        """Start the bridge communication"""
        self._running = True
        
        # Start socket server
        server_thread = threading.Thread(target=self._socket_server)
        server_thread.daemon = True
        server_thread.start()
        self._threads.append(server_thread)
        
        # Start message processor
        processor_thread = threading.Thread(target=self._process_messages)
        processor_thread.daemon = True
        processor_thread.start()
        self._threads.append(processor_thread)
        
        print(f"Unity bridge started on port {self.port}")
    
    def stop(self):
        """Stop the bridge communication"""
        self._running = False
        if self.socket:
            self.socket.close()
    
    def _socket_server(self):
        """Socket server for Unity communication"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', self.port))
        server_socket.listen(1)
        
        print(f"Waiting for Unity connection on port {self.port}")
        
        while self._running:
            try:
                client_socket, address = server_socket.accept()
                self.socket = client_socket
                self.is_connected = True
                print(f"Unity connected from {address}")
                
                # Handle client messages
                while self._running and self.is_connected:
                    try:
                        # Read message length (4 bytes)
                        length_data = client_socket.recv(4)
                        if not length_data:
                            break
                        
                        message_length = struct.unpack('I', length_data)[0]
                        
                        # Read message data
                        message_data = b''
                        while len(message_data) < message_length:
                            chunk = client_socket.recv(message_length - len(message_data))
                            if not chunk:
                                break
                            message_data += chunk
                        
                        # Process message
                        message = json.loads(message_data.decode('utf-8'))
                        message['timestamp'] = time.time()
                        self.inbound_queue.put(message)
                        
                    except Exception as e:
                        print(f"Error receiving message: {e}")
                        break
                
                self.is_connected = False
                client_socket.close()
                
            except Exception as e:
                if self._running:
                    print(f"Socket server error: {e}")
                break
        
        server_socket.close()
    
    def _process_messages(self):
        """Process inbound messages"""
        while self._running:
            try:
                # Process inbound messages
                try:
                    message = self.inbound_queue.get(timeout=0.1)
                    self._handle_message(message)
                except Empty:
                    pass
                
                # Send outbound messages
                try:
                    message = self.outbound_queue.get_nowait()
                    self._send_message(message)
                except Empty:
                    pass
                    
            except Exception as e:
                print(f"Message processing error: {e}")
    
    def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message from Unity"""
        start_time = time.time()
        
        message_type = message.get('type')
        if message_type in self.message_handlers:
            try:
                response = self.message_handlers[message_type](message)
                if response:
                    response['request_id'] = message.get('request_id')
                    self.send_message(response)
            except Exception as e:
                print(f"Error handling message {message_type}: {e}")
        
        # Track performance
        processing_time = (time.time() - start_time) * 1000  # ms
        self.message_times.append(processing_time)
        if len(self.message_times) > 1000:
            self.message_times.pop(0)
    
    def _send_message(self, message: Dict[str, Any]):
        """Send message to Unity"""
        if not self.is_connected:
            return False
        
        try:
            # Serialize message
            message_data = json.dumps(message).encode('utf-8')
            message_length = len(message_data)
            
            # Send length header + data
            length_header = struct.pack('I', message_length)
            self.socket.send(length_header + message_data)
            return True
            
        except Exception as e:
            print(f"Error sending message: {e}")
            return False
    
    def send_message(self, message: Dict[str, Any]):
        """Queue message for sending"""
        self.outbound_queue.put(message)
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type] = handler
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.message_times:
            return {}
        
        return {
            'avg_latency_ms': sum(self.message_times) / len(self.message_times),
            'max_latency_ms': max(self.message_times),
            'min_latency_ms': min(self.message_times),
            'total_messages': len(self.message_times)
        }

# Unity message handlers
class UnityMessageHandlers:
    """Handlers for different Unity message types"""
    
    def __init__(self, world_generator, asset_generator):
        self.world_generator = world_generator
        self.asset_generator = asset_generator
    
    def handle_generate_terrain(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle terrain generation request"""
        try:
            # Extract parameters
            size = message.get('size', [64, 64])
            biome = message.get('biome', 'forest')
            seed = message.get('seed', None)
            
            # Check cache first
            cache_key = f"terrain_{size[0]}x{size[1]}_{biome}_{seed}"
            if cache_key in self.request_cache:
                return self.request_cache[cache_key]
            
            # Generate terrain
            terrain_data = self.world_generator.generate_terrain(size, biome, seed)
            
            response = {
                'type': 'terrain_generated',
                'data': terrain_data.tolist(),
                'metadata': {
                    'size': size,
                    'biome': biome,
                    'generation_time': time.time()
                }
            }
            
            # Cache response
            self.request_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Terrain generation failed: {str(e)}"
            }
    
    def handle_generate_asset(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle asset generation request"""
        try:
            asset_type = message.get('asset_type', 'texture')
            style = message.get('style', 'forest')
            resolution = message.get('resolution', [256, 256])
            
            # Generate asset
            asset_data = self.asset_generator.generate_asset(asset_type, style, resolution)
            
            return {
                'type': 'asset_generated',
                'asset_type': asset_type,
                'data': asset_data,
                'metadata': {
                    'style': style,
                    'resolution': resolution,
                    'generation_time': time.time()
                }
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Asset generation failed: {str(e)}"
            }