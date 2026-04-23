"""
comms.py — Realistic Mesh Network Communications
Simulates lossy, delayed, bandwidth-limited drone-to-drone communication.
Replaces the simple CommChannel with a proper network model.

Key features:
    - Configurable packet loss (distance-based + random)
    - Message latency (propagation + jitter)
    - Bandwidth limiting (messages/sec per link)
    - Link quality tracking (RSSI simulation)
    - Network partitioning detection
    - Message prioritization (emergency > formation > telemetry)
    - Heartbeat-based neighbor discovery with timeout
"""

import time
import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Deque
from collections import deque

from drone import dist


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class NetworkConfig:
    max_range: float = 100.0          # meters, hard cutoff
    effective_range: float = 70.0     # meters, beyond this loss increases sharply
    base_packet_loss: float = 0.02    # 2% base loss even at close range
    latency_base: float = 0.005       # 5ms base latency
    latency_jitter: float = 0.010     # up to 10ms random jitter
    bandwidth_limit: int = 50         # max messages per drone per second
    heartbeat_interval: float = 0.5   # seconds between heartbeats
    neighbor_timeout: float = 2.0     # seconds before neighbor considered lost
    message_ttl: int = 3              # max hops for relayed messages
    interference_factor: float = 0.0  # 0-1, simulates jamming/EMI


# ═══════════════════════════════════════════════════════════════════
# Message Types with Priority
# ═══════════════════════════════════════════════════════════════════

class MessagePriority(IntEnum):
    EMERGENCY = 0     # always delivered first
    COMMAND = 1       # formation commands, waypoints
    STATE = 2         # position updates, consensus
    TELEMETRY = 3     # sensor data, battery
    HEARTBEAT = 4     # keep-alive


@dataclass
class NetMessage:
    sender_id: str
    msg_type: str
    payload: Any
    priority: MessagePriority = MessagePriority.STATE
    target_id: Optional[str] = None   # None = broadcast
    timestamp: float = 0.0
    hop_count: int = 0
    ttl: int = 3
    msg_id: int = 0                   # dedup

    def __lt__(self, other):
        return self.priority < other.priority


# ═══════════════════════════════════════════════════════════════════
# Link Quality Model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LinkState:
    """Tracks quality of a single drone-to-drone link."""
    neighbor_id: str
    last_heard: float = 0.0
    rssi: float = 1.0                # 0-1, signal strength
    packet_loss_rate: float = 0.0    # observed loss over window
    latency_avg: float = 0.0        # observed avg latency
    messages_received: int = 0
    messages_lost: int = 0
    is_alive: bool = True

    @property
    def quality(self) -> float:
        """Overall link quality 0-1."""
        if not self.is_alive:
            return 0.0
        return self.rssi * (1.0 - self.packet_loss_rate)


# ═══════════════════════════════════════════════════════════════════
# Mesh Network
# ═══════════════════════════════════════════════════════════════════

class MeshNetwork:
    """
    Realistic drone mesh network. Drop-in replacement for CommChannel.

    Each drone has:
        - A delivery queue (messages arrive with delay)
        - Link state table (who can I hear, how well)
        - Bandwidth counter (rate limiting)
        - Heartbeat timer

    Usage:
        net = MeshNetwork(NetworkConfig())
        net.register("drone_001", callback)
        net.update_position("drone_001", pos)
        net.send(NetMessage(...))
        net.tick(dt)  # must call every sim tick to process queues
    """

    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
        self._callbacks: Dict[str, List[Callable]] = {}
        self._positions: Dict[str, np.ndarray] = {}
        self._delivery_queue: Deque = deque()  # (deliver_at_time, message)
        self._link_states: Dict[str, Dict[str, LinkState]] = {}  # drone_id -> {neighbor_id -> LinkState}
        self._bandwidth_counters: Dict[str, int] = {}  # messages sent this second
        self._heartbeat_timers: Dict[str, float] = {}
        self._msg_counter: int = 0
        self._seen_msgs: Dict[str, set] = {}  # drone_id -> set of msg_ids seen
        self._sim_time: float = 0.0

        # Metrics
        self.total_sent: int = 0
        self.total_delivered: int = 0
        self.total_dropped: int = 0

    def register(self, drone_id: str, callback: Callable):
        self._callbacks.setdefault(drone_id, []).append(callback)
        self._link_states[drone_id] = {}
        self._bandwidth_counters[drone_id] = 0
        self._heartbeat_timers[drone_id] = 0.0
        self._seen_msgs[drone_id] = set()

    def unregister(self, drone_id: str):
        self._callbacks.pop(drone_id, None)
        self._link_states.pop(drone_id, None)
        self._bandwidth_counters.pop(drone_id, None)
        self._seen_msgs.pop(drone_id, None)
        self._positions.pop(drone_id, None)
        # Purge from other drones' link states so it doesn't appear as neighbor
        for links in self._link_states.values():
            links.pop(drone_id, None)

    def update_position(self, drone_id: str, pos: np.ndarray):
        self._positions[drone_id] = pos.copy()

    # ── Sending ──

    def send(self, msg: NetMessage):
        """Queue a message for delivery with realistic network effects."""
        msg.timestamp = self._sim_time
        msg.msg_id = self._msg_counter
        self._msg_counter += 1
        self.total_sent += 1

        sender_pos = self._positions.get(msg.sender_id)
        if sender_pos is None:
            return

        # Rate limit check
        if self._bandwidth_counters.get(msg.sender_id, 0) >= self.config.bandwidth_limit:
            self.total_dropped += 1
            return

        self._bandwidth_counters[msg.sender_id] = self._bandwidth_counters.get(msg.sender_id, 0) + 1

        if msg.target_id:
            self._try_deliver(msg, msg.target_id, sender_pos)
        else:
            for drone_id in self._callbacks:
                if drone_id == msg.sender_id:
                    continue
                self._try_deliver(msg, drone_id, sender_pos)

    def _try_deliver(self, msg: NetMessage, target_id: str, sender_pos: np.ndarray):
        """Apply network effects and queue for delayed delivery."""
        target_pos = self._positions.get(target_id)
        if target_pos is None:
            return

        d = dist(sender_pos, target_pos)
        cfg = self.config

        # Hard range cutoff
        if d > cfg.max_range:
            self.total_dropped += 1
            self._record_loss(msg.sender_id, target_id)
            return

        # Distance-based packet loss
        loss_prob = cfg.base_packet_loss
        if d > cfg.effective_range:
            excess = (d - cfg.effective_range) / (cfg.max_range - cfg.effective_range)
            loss_prob += excess ** 2 * 0.6  # steep falloff beyond effective range

        # Interference / jamming
        loss_prob += cfg.interference_factor * 0.5

        # Emergency messages get priority — lower loss
        if msg.priority == MessagePriority.EMERGENCY:
            loss_prob *= 0.3

        if np.random.random() < loss_prob:
            self.total_dropped += 1
            self._record_loss(msg.sender_id, target_id)
            return

        # Latency
        latency = cfg.latency_base + np.random.random() * cfg.latency_jitter
        latency += d / 300000.0  # speed-of-light propagation (tiny but realistic)

        deliver_at = self._sim_time + latency
        self._delivery_queue.append((deliver_at, target_id, msg))

    def _record_loss(self, sender_id: str, target_id: str):
        links = self._link_states.get(target_id, {})
        if sender_id in links:
            links[sender_id].messages_lost += 1

    # ── Tick Processing ──

    def tick(self, dt: float):
        """Process delivery queue and heartbeats. Call every sim tick."""
        self._sim_time += dt

        # Reset bandwidth counters every second
        if int(self._sim_time) != int(self._sim_time - dt):
            self._bandwidth_counters = {k: 0 for k in self._callbacks}

        # Deliver queued messages
        while self._delivery_queue and self._delivery_queue[0][0] <= self._sim_time:
            _, target_id, msg = self._delivery_queue.popleft()

            if target_id not in self._callbacks:
                continue

            # Dedup
            seen = self._seen_msgs.get(target_id, set())
            if msg.msg_id in seen:
                continue
            seen.add(msg.msg_id)
            if len(seen) > 1000:
                seen.clear()

            # Deliver
            for cb in self._callbacks.get(target_id, []):
                cb(msg)
            self.total_delivered += 1

            # Update link state
            self._update_link(target_id, msg.sender_id)

        # Heartbeat processing
        for drone_id in list(self._callbacks.keys()):
            self._heartbeat_timers.setdefault(drone_id, 0.0)
            self._heartbeat_timers[drone_id] += dt
            if self._heartbeat_timers[drone_id] >= self.config.heartbeat_interval:
                self._heartbeat_timers[drone_id] = 0.0
                self.send(NetMessage(
                    sender_id=drone_id,
                    msg_type="heartbeat",
                    payload={"time": self._sim_time},
                    priority=MessagePriority.HEARTBEAT,
                ))

        # Timeout dead neighbors
        for drone_id, links in self._link_states.items():
            for nid, link in links.items():
                if self._sim_time - link.last_heard > self.config.neighbor_timeout:
                    link.is_alive = False

    def _update_link(self, receiver_id: str, sender_id: str):
        links = self._link_states.setdefault(receiver_id, {})
        if sender_id not in links:
            links[sender_id] = LinkState(neighbor_id=sender_id)

        link = links[sender_id]
        link.last_heard = self._sim_time
        link.is_alive = True
        link.messages_received += 1

        # Update RSSI based on distance
        pos_r = self._positions.get(receiver_id)
        pos_s = self._positions.get(sender_id)
        if pos_r is not None and pos_s is not None:
            d = dist(pos_r, pos_s)
            link.rssi = max(0.0, 1.0 - (d / self.config.max_range) ** 2)

        # Update observed packet loss
        total = link.messages_received + link.messages_lost
        if total > 0:
            link.packet_loss_rate = link.messages_lost / total

    # ── Queries ──

    def get_neighbors(self, drone_id: str) -> List[str]:
        """Get IDs of drones this drone can currently communicate with."""
        links = self._link_states.get(drone_id, {})
        return [nid for nid, link in links.items()
                if link.is_alive and nid in self._callbacks]

    def get_link_quality(self, drone_id: str, neighbor_id: str) -> float:
        """Get link quality 0-1 between two drones."""
        links = self._link_states.get(drone_id, {})
        if neighbor_id in links:
            return links[neighbor_id].quality
        return 0.0

    def get_network_graph(self) -> Dict[str, List[str]]:
        """Get current connectivity graph."""
        graph = {}
        for drone_id in self._callbacks:
            graph[drone_id] = self.get_neighbors(drone_id)
        return graph

    def is_partitioned(self) -> bool:
        """Check if the network has split into disconnected groups."""
        if not self._callbacks:
            return False
        graph = self.get_network_graph()
        visited = set()
        start = next(iter(graph))
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        return len(visited) < len(graph)

    def get_partitions(self) -> List[List[str]]:
        """Get list of connected components."""
        graph = self.get_network_graph()
        visited = set()
        partitions = []
        for node in graph:
            if node in visited:
                continue
            component = []
            queue = [node]
            while queue:
                n = queue.pop(0)
                if n in visited:
                    continue
                visited.add(n)
                component.append(n)
                for neighbor in graph.get(n, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            partitions.append(component)
        return partitions

    @property
    def delivery_rate(self) -> float:
        if self.total_sent == 0:
            return 1.0
        return min(1.0, self.total_delivered / max(self.total_sent, 1))

    def stats(self) -> Dict:
        return {
            "sent": self.total_sent,
            "delivered": self.total_delivered,
            "dropped": self.total_dropped,
            "delivery_rate": f"{self.delivery_rate:.1%}",
            "partitioned": self.is_partitioned(),
            "partitions": len(self.get_partitions()),
        }
