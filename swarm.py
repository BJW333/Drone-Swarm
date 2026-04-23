"""
swarm.py — Swarm Manager
Central coordinator: spawns drones, runs the tick loop, dispatches AI systems,
manages obstacles, handles comms, loads plugins, and runs simulation.
This is the main file you interact with.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from drone import DroneAgent, DroneMode, PhysicsConfig, dist
from avoidance import ObstacleAvoidanceSystem, AvoidanceConfig
from formation import (FormationController, FormationType, BoidsConfig,
                       BoidsBehavior)
from comms import MeshNetwork, NetworkConfig, NetMessage, MessagePriority
from consensus import ConsensusEngine, ConsensusConfig, DroneRole
from missions import MissionPlanner
from metrics import MetricsCollector


# =============================================================================
# Communication Layer
# =============================================================================

class MessageType(Enum):
    POSITION = "position"
    OBSTACLE_ALERT = "obstacle_alert"
    FORMATION_CMD = "formation_cmd"
    EMERGENCY_STOP = "emergency_stop"
    HEARTBEAT = "heartbeat"
    CUSTOM = "custom"


@dataclass
class Message:
    sender_id: str
    msg_type: MessageType
    payload: Any
    target_id: Optional[str] = None  # None = broadcast
    timestamp: float = field(default_factory=time.time)


class CommChannel:
    """
    Inter-drone communication. Broadcast or direct messages with range limiting.
    For hardware, subclass and swap the transport (radio, wifi, etc).
    """

    def __init__(self, max_range: float = 100.0):
        self.max_range = max_range
        self._subs: Dict[str, List[Callable]] = {}
        self._positions: Dict[str, np.ndarray] = {}

    def register(self, drone_id: str, callback: Callable[[Message], None]):
        self._subs.setdefault(drone_id, []).append(callback)

    def unregister(self, drone_id: str):
        self._subs.pop(drone_id, None)

    def update_position(self, drone_id: str, pos: np.ndarray):
        self._positions[drone_id] = pos

    def send(self, msg: Message):
        if msg.target_id:
            for cb in self._subs.get(msg.target_id, []):
                cb(msg)
        else:
            sender_pos = self._positions.get(msg.sender_id)
            for did, callbacks in self._subs.items():
                if did == msg.sender_id:
                    continue
                if sender_pos is not None and did in self._positions:
                    if dist(sender_pos, self._positions[did]) > self.max_range:
                        continue
                for cb in callbacks:
                    cb(msg)

    def get_neighbors(self, drone_id: str) -> List[str]:
        pos = self._positions.get(drone_id)
        if pos is None:
            return [k for k in self._subs if k != drone_id]
        return [k for k in self._subs if k != drone_id
                and (k not in self._positions
                     or dist(pos, self._positions[k]) <= self.max_range)]


# =============================================================================
# Plugin System
# =============================================================================

class DronePlugin:
    """
    Base class for plugins. Subclass and override on_tick / on_event.

    Example:
        class GeofencePlugin(DronePlugin):
            name = "geofence"
            def on_tick(self, drones, swarm_state):
                for d in drones.values():
                    if out_of_bounds(d.position):
                        d.add_plugin_vector("geofence", push_back)
    """
    name: str = "unnamed"
    version: str = "1.0"
    priority: int = 0
    enabled: bool = True

    def on_load(self, config: Dict):
        pass

    def on_tick(self, drones: Dict[str, DroneAgent], swarm_state: Dict):
        pass

    def on_event(self, event_type: str, data: Any):
        pass

    def on_unload(self):
        pass


class PluginManager:
    def __init__(self):
        self._plugins: Dict[str, DronePlugin] = {}

    def register(self, plugin: DronePlugin, config: Dict = None):
        self._plugins[plugin.name] = plugin
        plugin.on_load(config or {})

    def unregister(self, name: str):
        if name in self._plugins:
            self._plugins[name].on_unload()
            del self._plugins[name]

    def get(self, name: str) -> Optional[DronePlugin]:
        return self._plugins.get(name)

    def on_tick(self, drones, state):
        for p in sorted(self._plugins.values(), key=lambda p: -p.priority):
            if p.enabled:
                try:
                    p.on_tick(drones, state)
                except Exception as e:
                    print(f"[Plugin:{p.name}] Error: {e}")

    def on_event(self, event_type, data):
        for p in self._plugins.values():
            if p.enabled:
                p.on_event(event_type, data)

    def list(self) -> List[str]:
        return [f"{p.name} v{p.version} ({'on' if p.enabled else 'off'})"
                for p in self._plugins.values()]


# =============================================================================
# Built-in Plugins
# =============================================================================

class GeofencePlugin(DronePlugin):
    """Keeps drones inside a bounding box."""
    name = "geofence"
    priority = 10

    def __init__(self, min_b=None, max_b=None, strength=8.0):
        from drone import clamp_mag
        self.min_b = min_b if min_b is not None else np.array([-100, -100, 0])
        self.max_b = max_b if max_b is not None else np.array([100, 100, 50])
        self.strength = strength

    def on_tick(self, drones, state):
        from drone import clamp_mag
        margin = 5.0
        for d in drones.values():
            force = np.zeros(3)
            for ax in range(3):
                if d.position[ax] < self.min_b[ax] + margin:
                    force[ax] = self.strength * (1 - (d.position[ax] - self.min_b[ax]) / margin)
                elif d.position[ax] > self.max_b[ax] - margin:
                    force[ax] = -self.strength * (1 - (self.max_b[ax] - d.position[ax]) / margin)
            if np.linalg.norm(force) > 0.01:
                d.add_plugin_vector("geofence", clamp_mag(force, self.strength))


class TelemetryPlugin(DronePlugin):
    """Logs positions/velocities for analysis."""
    name = "telemetry"
    priority = -10

    def __init__(self, interval=30):
        self.interval = interval
        self.log: List[Dict] = []

    def on_tick(self, drones, state):
        if state["tick"] % self.interval == 0:
            self.log.append({
                "tick": state["tick"], "time": state["elapsed"],
                "drones": {did: {"pos": d.position.tolist(),
                                 "vel": d.velocity.tolist(),
                                 "mode": d.mode.value}
                           for did, d in drones.items()}
            })

    def save(self, path):
        import json
        with open(path, 'w') as f:
            json.dump(self.log, f, indent=2)


class BatteryPlugin(DronePlugin):
    """Simulates battery drain; forces landing when low."""
    name = "battery"
    priority = 5

    def __init__(self, drain_rate=0.02, low_pct=20.0):
        self.drain_rate = drain_rate
        self.low_pct = low_pct
        self.levels: Dict[str, float] = {}

    def on_tick(self, drones, state):
        for did, d in drones.items():
            self.levels.setdefault(did, 100.0)
            speed_factor = 1.0 + np.linalg.norm(d.velocity) / 10.0
            self.levels[did] = max(0, self.levels[did] - self.drain_rate * speed_factor)
            d.battery_pct = self.levels[did]
            if self.levels[did] < self.low_pct and d.mode not in (DroneMode.LANDING, DroneMode.IDLE):
                d.land()


# =============================================================================
# Obstacles
# =============================================================================

class Obstacle:
    def __init__(self, position: np.ndarray, radius: float, velocity: np.ndarray = None):
        self.position = np.array(position, dtype=float)
        self.radius = radius
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else np.zeros(3)

    def to_dict(self):
        return {"position": self.position, "radius": self.radius, "velocity": self.velocity}

    def update(self, dt):
        self.position += self.velocity * dt


# =============================================================================
# Swarm Manager
# =============================================================================

@dataclass
class SwarmConfig:
    num_drones: int = 10
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    boids: BoidsConfig = field(default_factory=BoidsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    comm_range: float = 100.0
    tick_rate: float = 30.0
    default_altitude: float = 10.0
    spawn_spacing: float = 3.0
    formation: FormationType = FormationType.NONE
    enable_boids: bool = True
    enable_avoidance: bool = True
    enable_consensus: bool = True
    enable_comms_sim: bool = True
    enable_metrics: bool = True


class SwarmManager:
    """
    Central swarm coordinator. This is your main entry point.

    Quick start:
        swarm = SwarmManager(SwarmConfig(num_drones=20))
        swarm.add_obstacle(np.array([20, 0, 10]), radius=4.0)
        swarm.takeoff_all()
        swarm.set_formation(FormationType.V_SHAPE)
        swarm.run(duration=30)

    Custom loop:
        while running:
            swarm.tick()
            positions = swarm.get_positions()
            # render, log, whatever
    """

    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        self.drones: Dict[str, DroneAgent] = {}
        self.obstacles: List[Obstacle] = []
        self.comm = CommChannel(self.config.comm_range)
        self.tick_count = 0
        self.elapsed_time = 0.0
        self.dt = 1.0 / self.config.tick_rate

        # AI systems
        self.avoidance = ObstacleAvoidanceSystem(self.config.avoidance)
        self.formation_ctrl = FormationController()
        self.boids = BoidsBehavior(self.config.boids)
        self.plugins = PluginManager()

        # New systems
        self.network = MeshNetwork(self.config.network) if self.config.enable_comms_sim else None
        self.consensus = ConsensusEngine(self.config.consensus) if self.config.enable_consensus else None
        self.mission_planner = MissionPlanner()
        self.metrics = MetricsCollector() if self.config.enable_metrics else None

        # Hooks
        self._on_tick: List[Callable] = []

        self._spawn_drones()

    def _spawn_drones(self):
        n = self.config.num_drones
        cols = max(1, int(np.ceil(np.sqrt(n))))
        sp = self.config.spawn_spacing
        for i in range(n):
            pos = np.array([
                (i % cols - (cols-1)/2) * sp,
                (i // cols - ((n//cols)-1)/2) * sp, 0.0
            ])
            did = f"drone_{i:03d}"
            drone = DroneAgent(did, self.config.physics, pos)
            drone.target_altitude = self.config.default_altitude
            self.comm.register(did, lambda m, d=drone: None)
            if self.network:
                self.network.register(did, lambda m, _did=did: self._on_net_message(_did, m))
                self.network.update_position(did, pos)
            self.drones[did] = drone

        # Initialize consensus after all drones spawned
        if self.consensus:
            self.consensus.init(self.drones, self.network or self.comm)

    def _on_net_message(self, drone_id: str, msg: NetMessage):
        """Route network messages to consensus engine."""
        if self.consensus:
            self.consensus.handle_message(drone_id, msg)

    # --- Drone management ---

    def add_drone(self, drone_id=None, position=None) -> DroneAgent:
        did = drone_id or f"drone_{len(self.drones):03d}"
        d = DroneAgent(did, self.config.physics, position or np.zeros(3))
        d.target_altitude = self.config.default_altitude
        self.drones[did] = d
        return d

    def remove_drone(self, drone_id: str):
        self.comm.unregister(drone_id)
        if self.network:
            self.network.unregister(drone_id)
        self.drones.pop(drone_id, None)

    # --- Obstacles ---

    def add_obstacle(self, position: np.ndarray, radius: float,
                     velocity: np.ndarray = None) -> Obstacle:
        obs = Obstacle(position, radius, velocity)
        self.obstacles.append(obs)
        return obs

    def clear_obstacles(self):
        self.obstacles.clear()

    # --- Swarm Commands ---

    def takeoff_all(self, altitude=None):
        for d in self.drones.values():
            d.takeoff(altitude or self.config.default_altitude)

    def land_all(self):
        for d in self.drones.values():
            d.land()

    def emergency_stop_all(self):
        for d in self.drones.values():
            d.emergency_stop()

    def set_formation(self, formation: FormationType, center=None, **params):
        if center is None:
            center = self.get_centroid()
        targets = self.formation_ctrl.compute(formation, len(self.drones), center, **params)
        for drone, target in zip(self.drones.values(), targets):
            drone.target = target
            drone.mode = DroneMode.FORMATION

    def send_all_to(self, position: np.ndarray):
        for d in self.drones.values():
            d.go_to(position)

    # --- Main Tick ---

    def tick(self):
        """One simulation step: sense → AI → consensus → missions → plugins → act → metrics."""
        dt = self.dt
        obs_dicts = [o.to_dict() for o in self.obstacles]

        for obs in self.obstacles:
            obs.update(dt)

        # Sense
        for d in self.drones.values():
            d.sense(obs_dicts, self.drones)
            self.comm.update_position(d.id, d.position)
            if self.network:
                self.network.update_position(d.id, d.position)

        # Mesh network tick (process message queues, heartbeats)
        if self.network:
            self.network.tick(dt)

        # Obstacle avoidance
        if self.config.enable_avoidance:
            for d in self.drones.values():
                d.set_avoidance_vector(self.avoidance.compute(d, obs_dicts, self.drones))

        # Boids flocking
        if self.config.enable_boids:
            drone_list = list(self.drones.values())
            for d, vec in zip(drone_list, self.boids.compute_all(drone_list)):
                d.add_plugin_vector("separation", vec)

        # Decentralized consensus
        if self.consensus:
            self.consensus.tick(self.drones, self.network or self.comm, dt)

        # Mission planner
        self.mission_planner.tick(self.drones, dt, self.elapsed_time)

        # Plugins
        state = self._state()
        self.plugins.on_tick(self.drones, state)

        # Act
        for d in self.drones.values():
            d.act(dt)

        # Metrics
        if self.metrics:
            self.metrics.record(self.drones, obs_dicts, self.network, self.consensus)

        # Callbacks
        for cb in self._on_tick:
            cb(self)

        self.tick_count += 1
        self.elapsed_time += dt

    def run(self, duration=None, max_ticks=None, realtime=False):
        """Run the loop for a duration or tick count."""
        tick = 0
        while True:
            if max_ticks and tick >= max_ticks:
                break
            if duration and self.elapsed_time >= duration:
                break
            t0 = time.time()
            self.tick()
            tick += 1
            if realtime:
                sleep = self.dt - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)

    # --- Queries ---

    def _state(self) -> Dict:
        return {
            "tick": self.tick_count,
            "elapsed": self.elapsed_time,
            "drone_count": len(self.drones),
            "obstacles": [o.to_dict() for o in self.obstacles],
        }

    def get_positions(self) -> Dict[str, np.ndarray]:
        return {did: d.position.copy() for did, d in self.drones.items()}

    def get_centroid(self) -> np.ndarray:
        if not self.drones:
            return np.zeros(3)
        return np.mean([d.position for d in self.drones.values()], axis=0)

    def on_tick(self, callback: Callable):
        self._on_tick.append(callback)

    def status(self) -> str:
        modes = {}
        for d in self.drones.values():
            modes[d.mode.value] = modes.get(d.mode.value, 0) + 1
        parts = [f"Swarm: {len(self.drones)} drones | Tick {self.tick_count} | "
                 f"Time {self.elapsed_time:.1f}s | Modes: {modes}"]
        if self.consensus:
            parts.append(f"  {self.consensus.status()}")
        if self.network:
            parts.append(f"  Net: {self.network.stats()}")
        if self.mission_planner.get_active_missions():
            parts.append(f"  {self.mission_planner.status()}")
        return "\n".join(parts)
