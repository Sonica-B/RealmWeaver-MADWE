# src/performance/optimization_framework.py
import time
import psutil
import GPUtil
import threading
from collections import defaultdict, deque
from typing import Dict, List, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Performance metrics structure"""
    fps: float
    frame_time_ms: float
    cpu_usage: float
    gpu_usage: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    agent_overhead_ms: float
    generation_time_ms: float

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, target_fps: float = 20.0):
        self.target_fps = target_fps
        self.target_frame_time = 1000.0 / target_fps  # ms
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)
        self.frame_times = deque(maxlen=100)
        self.generation_times = deque(maxlen=100)
        
        # Performance counters
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            time.sleep(0.1)  # Monitor every 100ms
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # CPU/Memory metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / (1024 * 1024)
        
        # GPU metrics
        gpu_usage = 0.0
        gpu_memory_usage_mb = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage = gpu.load * 100
                gpu_memory_usage_mb = gpu.memoryUsed
        except:
            pass
        
        # Frame time metrics
        avg_frame_time = np.mean(self.frame_times) if self.frame_times else 0
        avg_generation_time = np.mean(self.generation_times) if self.generation_times else 0
        
        return PerformanceMetrics(
            fps=self.current_fps,
            frame_time_ms=avg_frame_time,
            cpu_usage=cpu_usage,
            gpu_usage=gpu_usage,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_usage_mb=gpu_memory_usage_mb,
            agent_overhead_ms=0.0,  # To be filled by agent system
            generation_time_ms=avg_generation_time
        )
    
    def record_frame(self, frame_time_ms: float):
        """Record frame timing"""
        self.frame_times.append(frame_time_ms)
        self.frame_count += 1
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def record_generation_time(self, generation_time_ms: float):
        """Record generation timing"""
        self.generation_times.append(generation_time_ms)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get latest performance metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    def is_performance_acceptable(self) -> bool:
        """Check if performance meets targets"""
        if not self.metrics_history:
            return False
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
        avg_fps = np.mean([m.fps for m in recent_metrics])
        
        return avg_fps >= self.target_fps * 0.9  # 90% of target

class PerformanceOptimizer:
    """Automatic performance optimization system"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_strategies = {
            'reduce_quality': self._reduce_generation_quality,
            'reduce_agents': self._reduce_active_agents,
            'increase_cache': self._increase_caching,
            'reduce_resolution': self._reduce_asset_resolution
        }
        
        # Optimization state
        self.current_quality_level = 1.0
        self.max_active_agents = 10
        self.cache_size_multiplier = 1.0
        self.asset_resolution_multiplier = 1.0
        
    def optimize_performance(self):
        """Run automatic performance optimization"""
        metrics = self.monitor.get_current_metrics()
        
        if not self.monitor.is_performance_acceptable():
            # Performance is below target, apply optimizations
            self._apply_optimizations(metrics)
        elif metrics.fps > self.monitor.target_fps * 1.2:
            # Performance is well above target, we can increase quality
            self._increase_quality()
    
    def _apply_optimizations(self, metrics: PerformanceMetrics):
        """Apply performance optimizations based on bottlenecks"""
        # Identify primary bottleneck
        if metrics.gpu_usage > 90:
            # GPU bottleneck - reduce quality or resolution
            if self.current_quality_level > 0.5:
                self._reduce_generation_quality()
            else:
                self._reduce_asset_resolution()
        elif metrics.cpu_usage > 90:
            # CPU bottleneck - reduce agent count
            self._reduce_active_agents()
        elif metrics.memory_usage_mb > 6000:  # >6GB
            # Memory bottleneck - increase caching efficiency
            self._increase_caching()
    
    def _reduce_generation_quality(self):
        """Reduce AI generation quality for performance"""
        self.current_quality_level = max(0.3, self.current_quality_level * 0.8)
        print(f"Reduced generation quality to {self.current_quality_level:.2f}")
        
    def _reduce_active_agents(self):
        """Reduce number of active agents"""
        self.max_active_agents = max(3, int(self.max_active_agents * 0.8))
        print(f"Reduced max active agents to {self.max_active_agents}")
        
    def _increase_caching(self):
        """Increase caching to reduce generation load"""
        self.cache_size_multiplier = min(3.0, self.cache_size_multiplier * 1.2)
        print(f"Increased cache size multiplier to {self.cache_size_multiplier:.2f}")
        
    def _reduce_asset_resolution(self):
        """Reduce asset resolution for performance"""
        self.asset_resolution_multiplier = max(0.5, self.asset_resolution_multiplier * 0.8)
        print(f"Reduced asset resolution multiplier to {self.asset_resolution_multiplier:.2f}")
        
    def _increase_quality(self):
        """Gradually increase quality when performance allows"""
        if self.current_quality_level < 1.0:
            self.current_quality_level = min(1.0, self.current_quality_level * 1.05)
        elif self.max_active_agents < 10:
            self.max_active_agents = min(10, self.max_active_agents + 1)
        elif self.asset_resolution_multiplier < 1.0:
            self.asset_resolution_multiplier = min(1.0, self.asset_resolution_multiplier * 1.05)

# Usage example and integration
class MADWEPerformanceSystem:
    """Main performance system integration"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor(target_fps=20.0)
        self.optimizer = PerformanceOptimizer(self.monitor)
        
    def start(self):
        """Start performance monitoring and optimization"""
        self.monitor.start_monitoring()
        
        # Start optimization loop
        self._optimization_thread = threading.Thread(target=self._optimization_loop)
        self._optimization_thread.daemon = True
        self._optimization_thread.start()
        
    def _optimization_loop(self):
        """Main optimization loop"""
        while True:
            self.optimizer.optimize_performance()
            time.sleep(1.0)  # Optimize every second
            
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for performance dashboard"""
        metrics = self.monitor.get_current_metrics()
        
        return {
            'fps': metrics.fps,
            'target_fps': self.monitor.target_fps,
            'frame_time_ms': metrics.frame_time_ms,
            'cpu_usage': metrics.cpu_usage,
            'gpu_usage': metrics.gpu_usage,
            'memory_usage_gb': metrics.memory_usage_mb / 1024,
            'gpu_memory_usage_mb': metrics.gpu_memory_usage_mb,
            'quality_level': self.optimizer.current_quality_level,
            'active_agents': self.optimizer.max_active_agents,
            'performance_acceptable': self.monitor.is_performance_acceptable()
        }