"""
metrics.py — Swarm Performance Metrics & Evaluation
Quantitative measurements for swarm performance analysis.

Tracks:
    - Formation accuracy (how close drones are to target positions)
    - Obstacle clearance margins (minimum distance to obstacles over time)
    - Communication health (delivery rate, partition events, latency)
    - Swarm cohesion (spread, connectivity graph density)
    - Mission performance (completion rate, time-to-complete)
    - Resilience (response to drone loss, recovery time)
    - Per-drone telemetry (speed, acceleration, energy usage)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from drone import DroneAgent, dist


# ═══════════════════════════════════════════════════════════════════
# Metric Snapshots
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SwarmSnapshot:
    """One moment in time, all metrics."""
    tick: int = 0
    time: float = 0.0
    drone_count: int = 0
    # Formation
    formation_error: float = 0.0        # avg distance from target positions
    formation_max_error: float = 0.0    # worst drone
    # Obstacles
    min_obstacle_clearance: float = float('inf')  # closest any drone got to any obstacle
    clearance_violations: int = 0       # how many drones are within danger radius
    # Cohesion
    swarm_spread: float = 0.0           # max distance between any two drones
    avg_neighbor_dist: float = 0.0      # average nearest-neighbor distance
    connectivity: float = 0.0           # fraction of possible links that are active
    # Comms
    msg_delivery_rate: float = 1.0
    network_partitions: int = 1
    # Performance
    avg_speed: float = 0.0
    max_speed: float = 0.0
    total_distance_traveled: float = 0.0
    avg_battery: float = 100.0
    drones_active: int = 0
    drones_isolated: int = 0


# ═══════════════════════════════════════════════════════════════════
# Metrics Collector
# ═══════════════════════════════════════════════════════════════════

class MetricsCollector:
    """
    Collects swarm performance metrics every tick.

    Usage:
        metrics = MetricsCollector()
        # each tick:
        metrics.record(drones, obstacles, network, consensus)
        # after sim:
        metrics.summary()
        metrics.export("metrics.json")
    """

    def __init__(self, record_interval: int = 10):
        self.interval = record_interval
        self.history: List[SwarmSnapshot] = []
        self._tick_count: int = 0
        self._prev_positions: Dict[str, np.ndarray] = {}
        self._total_distances: Dict[str, float] = {}
        self._drone_loss_events: List[Dict] = []
        self._prev_drone_count: int = 0

    def record(self, drones: Dict[str, DroneAgent],
               obstacles: List[Dict],
               network=None, consensus=None):
        """Record metrics for this tick."""
        self._tick_count += 1

        # Track distance traveled
        for did, drone in drones.items():
            if did in self._prev_positions:
                d = dist(drone.position, self._prev_positions[did])
                self._total_distances[did] = self._total_distances.get(did, 0.0) + d
            self._prev_positions[did] = drone.position.copy()

        # Detect drone loss
        if len(drones) < self._prev_drone_count:
            self._drone_loss_events.append({
                "tick": self._tick_count,
                "lost": self._prev_drone_count - len(drones),
            })
        self._prev_drone_count = len(drones)

        # Only snapshot at intervals
        if self._tick_count % self.interval != 0:
            return

        snap = SwarmSnapshot(
            tick=self._tick_count,
            time=self._tick_count / 30.0,  # assuming 30Hz
            drone_count=len(drones),
        )

        positions = [d.position for d in drones.values()]
        velocities = [d.velocity for d in drones.values()]

        if not positions:
            self.history.append(snap)
            return

        # ── Formation Error ──
        errors = []
        for d in drones.values():
            if d.target is not None:
                errors.append(dist(d.position, d.target))
        if errors:
            snap.formation_error = float(np.mean(errors))
            snap.formation_max_error = float(np.max(errors))

        # ── Obstacle Clearance ──
        min_clear = float('inf')
        violations = 0
        for d in drones.values():
            for obs in obstacles:
                obs_pos = obs["position"] if isinstance(obs, dict) else obs.position
                obs_r = obs["radius"] if isinstance(obs, dict) else obs.radius
                clearance = dist(d.position, obs_pos) - obs_r
                min_clear = min(min_clear, clearance)
                if clearance < 2.0:
                    violations += 1
        snap.min_obstacle_clearance = min_clear if min_clear != float('inf') else 0.0
        snap.clearance_violations = violations

        # ── Cohesion ──
        if len(positions) > 1:
            pos_arr = np.array(positions)
            # Spread = max pairwise distance
            max_spread = 0.0
            nn_dists = []
            for i in range(len(pos_arr)):
                min_nn = float('inf')
                for j in range(len(pos_arr)):
                    if i == j:
                        continue
                    d = dist(pos_arr[i], pos_arr[j])
                    max_spread = max(max_spread, d)
                    min_nn = min(min_nn, d)
                nn_dists.append(min_nn)
            snap.swarm_spread = max_spread
            snap.avg_neighbor_dist = float(np.mean(nn_dists))

        # ── Connectivity ──
        if network:
            graph = network.get_network_graph()
            n_drones = len(drones)
            total_possible = n_drones * (n_drones - 1) if n_drones > 1 else 1
            actual_links = sum(len(neighbors) for neighbors in graph.values())
            snap.connectivity = min(1.0, actual_links / total_possible)
            snap.msg_delivery_rate = network.delivery_rate
            snap.network_partitions = len(network.get_partitions())

        # ── Speed / Energy ──
        speeds = [np.linalg.norm(v) for v in velocities]
        snap.avg_speed = float(np.mean(speeds))
        snap.max_speed = float(np.max(speeds))
        snap.total_distance_traveled = sum(self._total_distances.values())
        snap.avg_battery = float(np.mean([d.battery_pct for d in drones.values()]))

        # ── Active/Isolated counts ──
        snap.drones_active = len(drones)
        if consensus:
            snap.drones_isolated = len(consensus.get_isolated())

        self.history.append(snap)

    # ── Analysis ──

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate metrics over entire simulation."""
        if not self.history:
            return {"error": "no data"}

        return {
            "duration": f"{self.history[-1].time:.1f}s",
            "total_ticks": self._tick_count,
            "snapshots": len(self.history),
            "formation": {
                "avg_error": f"{np.mean([s.formation_error for s in self.history]):.2f}m",
                "max_error": f"{max(s.formation_max_error for s in self.history):.2f}m",
                "final_error": f"{self.history[-1].formation_error:.2f}m",
            },
            "safety": {
                "min_obstacle_clearance": f"{min(s.min_obstacle_clearance for s in self.history):.2f}m",
                "avg_clearance_violations": f"{np.mean([s.clearance_violations for s in self.history]):.1f}",
                "total_violation_ticks": sum(1 for s in self.history if s.clearance_violations > 0),
            },
            "cohesion": {
                "avg_spread": f"{np.mean([s.swarm_spread for s in self.history]):.1f}m",
                "avg_neighbor_dist": f"{np.mean([s.avg_neighbor_dist for s in self.history]):.1f}m",
                "avg_connectivity": f"{np.mean([s.connectivity for s in self.history]):.1%}",
            },
            "comms": {
                "avg_delivery_rate": f"{np.mean([s.msg_delivery_rate for s in self.history]):.1%}",
                "partition_events": sum(1 for s in self.history if s.network_partitions > 1),
            },
            "performance": {
                "avg_speed": f"{np.mean([s.avg_speed for s in self.history]):.1f} m/s",
                "total_distance": f"{self.history[-1].total_distance_traveled:.0f}m",
                "final_battery": f"{self.history[-1].avg_battery:.0f}%",
            },
            "resilience": {
                "drone_loss_events": len(self._drone_loss_events),
                "max_isolated": max((s.drones_isolated for s in self.history), default=0),
            },
        }

    def export(self, filepath: str):
        """Export full metric history to JSON."""
        import json
        data = {
            "summary": self.summary(),
            "history": [
                {
                    "tick": s.tick, "time": round(s.time, 2),
                    "drones": s.drone_count,
                    "formation_error": round(s.formation_error, 2),
                    "min_clearance": round(s.min_obstacle_clearance, 2),
                    "spread": round(s.swarm_spread, 1),
                    "connectivity": round(s.connectivity, 3),
                    "delivery_rate": round(s.msg_delivery_rate, 3),
                    "avg_speed": round(s.avg_speed, 2),
                    "battery": round(s.avg_battery, 1),
                    "isolated": s.drones_isolated,
                }
                for s in self.history
            ],
            "drone_loss_events": self._drone_loss_events,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def print_summary(self):
        """Pretty-print the summary."""
        s = self.summary()
        print(f"\n{'='*60}")
        print(f"  SWARM PERFORMANCE REPORT  ({s['duration']}, {s['total_ticks']} ticks)")
        print(f"{'='*60}")
        for category, metrics in s.items():
            if isinstance(metrics, dict):
                print(f"\n  {category.upper()}:")
                for k, v in metrics.items():
                    print(f"    {k:.<35} {v}")
        print(f"{'='*60}\n")
