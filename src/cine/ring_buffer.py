import threading
import time
from typing import Optional, Tuple, Any


class FramePacket:
    """Carries a single frame payload through the cine ring buffer."""
    def __init__(self, frame_id: int, frame_img: Any, t_capture: float):
        self.frame_id = frame_id
        self.image = frame_img
        self.t_capture = t_capture


class CineRingBuffer:
    """
    Thread-safe bounded ring buffer with drop-oldest policy.
    When full, pushing a new frame overwrites the oldest unconsumed frame
    and increments the dropped_count.
    """
    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0  # Write pointer
        self.tail = 0  # Read pointer
        self.count = 0

        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)

        self.produced_count = 0
        self.processed_count = 0
        self.dropped_count = 0
        self.is_closed = False

    def push(self, packet: FramePacket) -> bool:
        """
        Push a new packet to the buffer.
        If buffer is full, drop the oldest frame (drop-oldest policy).
        """
        with self.lock:
            if self.is_closed:
                return False

            self.produced_count += 1
            was_dropped = False

            if self.count == self.capacity:
                # Buffer is full: overwrite oldest frame at tail
                self.tail = (self.tail + 1) % self.capacity
                self.dropped_count += 1
                self.count -= 1
                was_dropped = True

            self.buffer[self.head] = packet
            self.head = (self.head + 1) % self.capacity
            self.count += 1

            self.not_empty.notify()
            return was_dropped

    def pop_latest(self, timeout: float = 0.5) -> Optional[Tuple[FramePacket, float]]:
        """
        Pop the newest available frame (drop-oldest policy).
        Any unconsumed older frames in the buffer are dropped.
        Returns (packet, t_dequeue).
        """
        with self.not_empty:
            while self.count == 0 and not self.is_closed:
                if not self.not_empty.wait(timeout=timeout):
                    return None

            if self.count == 0 and self.is_closed:
                return None

            t_dequeue = time.perf_counter()
            # Drop-oldest newest-first consumption:
            # The newest frame is at index (self.head - 1) % self.capacity
            newest_idx = (self.head - 1) % self.capacity
            packet = self.buffer[newest_idx]
            self.buffer[newest_idx] = None

            # Any remaining unconsumed older frames in buffer are dropped
            stale_dropped = self.count - 1
            if stale_dropped > 0:
                curr = self.tail
                while curr != newest_idx:
                    self.buffer[curr] = None
                    curr = (curr + 1) % self.capacity
                self.dropped_count += stale_dropped

            self.count = 0
            self.tail = self.head
            self.processed_count += 1

            return packet, t_dequeue

    def close(self):
        """Signal that no more frames will be produced."""
        with self.lock:
            self.is_closed = True
            self.not_empty.notify_all()

    def get_stats(self) -> dict:
        """Return accounting counts to verify invariant: produced == processed + dropped."""
        with self.lock:
            return {
                "produced": self.produced_count,
                "processed": self.processed_count,
                "dropped": self.dropped_count,
                "in_flight": self.count,
                "capacity": self.capacity,
            }
