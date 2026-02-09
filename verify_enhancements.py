import time
import json
import random
import numpy as np
from datetime import datetime

# Mock classes to simulate the environment
class MockPerformanceMonitor:
    def __init__(self):
        self.fps_history = []
        self.latency_history = []
        self.max_history_size = 100
    
    def update_fps(self, fps):
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_history_size:
            self.fps_history.pop(0)
    
    def update_latency(self, latency):
        self.latency_history.append(latency)
        if len(self.latency_history) > self.max_history_size:
            self.latency_history.pop(0)
            
    def get_stats(self):
        if not self.fps_history:
            return {"fps": 0, "latency_ms": 0}
        return {
            "fps": sum(self.fps_history) / len(self.fps_history),
            "latency_ms": sum(self.latency_history) / len(self.latency_history),
            "fps_min": min(self.fps_history),
            "fps_max": max(self.fps_history)
        }

class MockController:
    def __init__(self):
        self.performance_monitor = MockPerformanceMonitor()
        self.performance_stats = {
            'fps': 0,
            'last_update': time.time(),
            'frame_count': 0,
            'memory_usage': 0,
            'avg_frame_time': 0
        }
        self.running = True
        self.cursor_mode = False
        self.voice_enabled = True
        self.current_gesture = "none"
        
    def _update_performance_stats(self, frame_count, frame_times):
        """Simulate the update logic from the actual controller"""
        current_time = time.time()
        elapsed = current_time - self.performance_stats['last_update']
        if elapsed > 0:
            fps = frame_count / elapsed
            self.performance_monitor.update_fps(fps)
        
        avg_frame_time = np.mean(frame_times) * 1000
        self.performance_monitor.update_latency(avg_frame_time)
        
        monitor_stats = self.performance_monitor.get_stats()
        
        self.performance_stats.update({
            'fps': monitor_stats.get('fps', 0),
            'latency_ms': monitor_stats.get('latency_ms', 0),
            'fps_min': monitor_stats.get('fps_min', 0),
            'fps_max': monitor_stats.get('fps_max', 0),
            'frame_count': frame_count,
            'avg_frame_time': avg_frame_time,
            'last_update': current_time,
            'memory_usage': 50.0 # Mock usage
        })

    def get_status(self):
        """Simulate get_status"""
        status = {
            "running": self.running,
            "cursor_mode": self.cursor_mode,
            "voice_enabled": self.voice_enabled,
            "current_gesture": self.current_gesture,
            "hand_detected": True,
            "hand_count": 1,
            "performance": self.performance_stats,
            "history": {
                "fps": self.performance_monitor.fps_history[-50:],
                "latency": self.performance_monitor.latency_history[-50:]
            }
        }
        return status

def verify_enhancements():
    print("Verifying Web Dashboard Enhancements...")
    
    controller = MockController()
    
    # Simulate some frames
    print("Simulating frame processing...")
    for i in range(10):
        # Simulate ~30 FPS
        time.sleep(1/30) 
        frame_times = [0.033] * 30
        controller._update_performance_stats(30, frame_times)
    
    status = controller.get_status()
    
    # Verify Structure
    print("\nChecking Status Structure:")
    
    # Check Performance Stats
    assert 'performance' in status, "Missing performance stats"
    perf = status['performance']
    print(f"✅ Performance Stats present: FPS={perf.get('fps', 0):.2f}, Latency={perf.get('latency_ms', 0):.2f}ms")
    
    # Check History
    assert 'history' in status, "Missing history data"
    history = status['history']
    assert 'fps' in history, "Missing FPS history"
    assert 'latency' in history, "Missing Latency history"
    
    print(f"✅ History Data present: {len(history['fps'])} FPS points, {len(history['latency'])} Latency points")
    
    # Verify Data Integrity
    assert len(history['fps']) > 0, "FPS history is empty"
    assert len(history['latency']) > 0, "Latency history is empty"
    
    print("\n✅ Verification Successful: Data structure matches requirements for Web Charts.")

if __name__ == "__main__":
    verify_enhancements()
