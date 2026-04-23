"""
drone.py — Individual Drone Agent
Physics, sensing, PD controller, and state management for a single drone.
This is the atomic unit of the swarm.
"""

import numpy as np
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


#Helpers for vector math and geometry

def normalize(v: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(v)
    return v / mag if mag > 1e-8 else np.zeros_like(v)

def clamp_mag(v: np.ndarray, max_mag: float) -> np.ndarray:
    mag = np.linalg.norm(v)
    return v * (max_mag / mag) if mag > max_mag and mag > 1e-8 else v

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


#Physics and dynamics configuration for drones
#Adjust these to simulate different drone models or hardware constraints

@dataclass
class PhysicsConfig:
    max_speed: float = 15.0
    max_acceleration: float = 8.0
    max_vertical_speed: float = 5.0
    drag: float = 0.1
    max_yaw_rate: float = 3.14


class DroneMode(Enum):
    IDLE = "idle"
    TAKEOFF = "takeoff"
    HOVER = "hover"
    NAVIGATE = "navigate"
    FORMATION = "formation"
    LANDING = "landing"
    EMERGENCY = "emergency"



#A single Drone Agent framework
#This is the core class that represents each drone in the swarm
#It handles its own state, physics, sensing, and control logic
#The swarm manager will create multiple instances of this class and call their methods each tick

class DroneAgent:
    """
    A single drone. Runs its own sense then decide then act loop.

    The swarm manager calls these each tick:
        drone.sense(obstacles, all_drones)
        drone.act(dt)

    External AI systems inject steering via:
        drone.set_avoidance_vector(vec)
        drone.set_formation_vector(vec)
        drone.add_plugin_vector(name, vec)
    """

    def __init__(self, drone_id: str, physics: PhysicsConfig = None,
                 start_position: np.ndarray = None):
        self.id = drone_id
        self.physics = physics or PhysicsConfig()

        #Kinematic state
        self.position = start_position.copy() if start_position is not None else np.zeros(3)
        self.velocity = np.zeros(3)
        self.heading = 0.0
        self.mode = DroneMode.IDLE
        self.home = self.position.copy()

        #Navigation
        self.waypoints: List[np.ndarray] = []
        self.waypoint_idx: int = 0
        self.target: Optional[np.ndarray] = None
        self.target_altitude: float = 10.0
        self.waypoint_radius: float = 2.0

        #Sensing (populated by sense())
        self.nearby_obstacles: List[Dict] = []
        self.nearby_drones: List[Dict] = []
        self.battery_pct: float = 100.0

        #Controller gains
        self.kp = 2.0
        self.kd = 1.5
        self.kp_alt = 3.0

        #Injected steering vectors from AI systems
        self._avoidance = np.zeros(3)
        self._formation = np.zeros(3)
        self._plugin_vectors: Dict[str, np.ndarray] = {}

        #Behavior blending weights
        self.weights = {
            "navigation": 1.0,
            "avoidance": 3.0,
            "formation": 0.8,
            "separation": 1.5,
        }

        #Arbitrary metadata (plugins can stash data here)
        self.meta: Dict[str, Any] = {}

    #--- Sensing ---

    def sense(self, obstacles: List[Dict], all_drones: Dict[str, 'DroneAgent'],
              detection_range: float = 30.0):
        """Gather sensor data from the world. Replace with real sensors for hardware."""
        self.nearby_obstacles = [
            obs for obs in obstacles
            if dist(self.position, obs["position"]) < detection_range + obs.get("radius", 0)
        ]
        self.nearby_drones = [
            {"id": d.id, "position": d.position.copy(), "velocity": d.velocity.copy()}
            for d in all_drones.values()
            if d.id != self.id and dist(self.position, d.position) < detection_range
        ]

    #--- Decision / Control ---

    def act(self, dt: float):
        """Compute blended acceleration, step physics, advance waypoints."""
        accel = self._decide()
        self._step_physics(accel, dt)
        self._advance_waypoints()
        #Reset per-tick injections
        self._avoidance = np.zeros(3)
        self._formation = np.zeros(3)
        self._plugin_vectors.clear()

    def _decide(self) -> np.ndarray:
        if self.mode in (DroneMode.IDLE, DroneMode.EMERGENCY):
            return np.zeros(3)
        if self.mode == DroneMode.TAKEOFF:
            return self._ctrl_takeoff()
        if self.mode == DroneMode.LANDING:
            return self._ctrl_land()

        #Blend navigation + all injected vectors
        w = self.weights
        total = w["navigation"] * self._ctrl_navigate()
        total += w["avoidance"] * self._avoidance
        total += w["formation"] * self._formation
        for name, vec in self._plugin_vectors.items():
            total += w.get(name, 1.0) * vec
        return total

    def _ctrl_navigate(self) -> np.ndarray:
        target = self.target
        if target is None and self.waypoints and self.waypoint_idx < len(self.waypoints):
            target = self.waypoints[self.waypoint_idx]
        if target is None:
            #Hover in place at target altitude
            hold = self.position.copy()
            hold[2] = self.target_altitude
            error = hold - self.position
            return clamp_mag(self.kp_alt * error - self.kd * self.velocity,
                             self.physics.max_acceleration * 0.5)

        error = target - self.position
        return clamp_mag(self.kp * error + self.kd * (-self.velocity),
                         self.physics.max_acceleration)

    def _ctrl_takeoff(self) -> np.ndarray:
        hold = self.position.copy()
        hold[2] = self.target_altitude
        accel = self.kp_alt * (hold - self.position) - self.kd * self.velocity
        if self.position[2] >= self.target_altitude - 0.5:
            self.mode = DroneMode.HOVER
        return clamp_mag(accel, self.physics.max_acceleration * 0.5)

    def _ctrl_land(self) -> np.ndarray:
        target = self.home.copy()
        target[2] = 0.0
        accel = 1.5 * (target - self.position) - 1.0 * self.velocity
        if self.position[2] < 0.3:
            self.mode = DroneMode.IDLE
            self.velocity = np.zeros(3)
        return clamp_mag(accel, self.physics.max_acceleration * 0.3)

    #--- Physics Integration ---

    def _step_physics(self, desired_accel: np.ndarray, dt: float):
        cfg = self.physics
        accel = clamp_mag(desired_accel, cfg.max_acceleration)
        accel += -cfg.drag * self.velocity  #drag

        self.velocity += accel * dt

        #Enforce speed limits
        h_speed = np.linalg.norm(self.velocity[:2])
        if h_speed > cfg.max_speed:
            self.velocity[:2] *= cfg.max_speed / h_speed
        self.velocity[2] = np.clip(self.velocity[2], -cfg.max_vertical_speed, cfg.max_vertical_speed)

        self.position += self.velocity * dt

        #Ground clamp
        if self.position[2] < 0:
            self.position[2] = 0
            self.velocity[2] = max(0, self.velocity[2])

        #Yaw toward velocity
        h = np.linalg.norm(self.velocity[:2])
        if h > 0.5:
            target_yaw = np.arctan2(self.velocity[1], self.velocity[0])
            diff = (target_yaw - self.heading + np.pi) % (2 * np.pi) - np.pi
            self.heading += np.clip(diff, -cfg.max_yaw_rate * dt, cfg.max_yaw_rate * dt)

    def _advance_waypoints(self):
        if not self.waypoints or self.waypoint_idx >= len(self.waypoints):
            return
        if dist(self.position, self.waypoints[self.waypoint_idx]) < self.waypoint_radius:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):
                self.mode = DroneMode.HOVER

    #--- Injection Setters (called by AI systems) ---

    def set_avoidance_vector(self, v: np.ndarray):
        self._avoidance = v

    def set_formation_vector(self, v: np.ndarray):
        self._formation = v

    def add_plugin_vector(self, name: str, v: np.ndarray):
        self._plugin_vectors[name] = v

    #--- Commands ---

    def takeoff(self, altitude: float = 10.0):
        self.target_altitude = altitude
        self.mode = DroneMode.TAKEOFF

    def land(self):
        self.mode = DroneMode.LANDING

    def go_to(self, position: np.ndarray):
        self.target = position.copy()
        self.mode = DroneMode.NAVIGATE

    def follow_waypoints(self, wps: List[np.ndarray]):
        self.waypoints = [w.copy() for w in wps]
        self.waypoint_idx = 0
        self.mode = DroneMode.NAVIGATE

    def emergency_stop(self):
        self.mode = DroneMode.EMERGENCY
        self.velocity = np.zeros(3)

    def __repr__(self):
        spd = np.linalg.norm(self.velocity)
        p = self.position
        return f"Drone({self.id}, {self.mode.value}, pos=[{p[0]:.1f},{p[1]:.1f},{p[2]:.1f}], spd={spd:.1f})"
