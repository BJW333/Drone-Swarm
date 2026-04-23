"""
missions.py — Mission Planning & Task Decomposition
High-level mission objectives decomposed into per-drone tasks.

Mission types:
    - AreaSearch:       Divide area into sectors, sweep systematically
    - PerimeterPatrol:  Drones orbit a perimeter at even spacing
    - TargetTrack:      Surround and follow a moving target
    - RelayChain:       Form a communication relay between two points
    - Escort:           Guard formation around a VIP point

Each mission runs independently and assigns waypoints/targets to drones.
Missions are resilient — they re-plan when drones are lost.
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from drone import DroneAgent, DroneMode, dist


# ═══════════════════════════════════════════════════════════════════
# Mission Base
# ═══════════════════════════════════════════════════════════════════

class MissionStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"


class Mission(ABC):
    """Base class for all missions."""

    name: str = "unnamed"
    status: MissionStatus = MissionStatus.PENDING
    assigned_drones: List[str] = []
    start_time: float = 0.0
    completion_pct: float = 0.0

    @abstractmethod
    def plan(self, drones: Dict[str, DroneAgent]):
        """Generate initial task assignments."""
        pass

    @abstractmethod
    def tick(self, drones: Dict[str, DroneAgent], dt: float, sim_time: float):
        """Update mission state each tick. Re-plan if needed."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return mission status for telemetry."""
        pass


# ═══════════════════════════════════════════════════════════════════
# Area Search
# ═══════════════════════════════════════════════════════════════════

class AreaSearchMission(Mission):
    """
    Divide a rectangular area into sectors and sweep them systematically.
    Each drone gets assigned sectors. If a drone is lost, its sectors
    are redistributed to remaining drones.

    Usage:
        mission = AreaSearchMission(
            center=np.array([0, 0, 10]),
            width=80, height=80, altitude=10
        )
        planner.assign_mission(mission, drone_ids)
    """

    name = "area_search"

    def __init__(self, center: np.ndarray, width: float = 80.0,
                 height: float = 80.0, altitude: float = 10.0,
                 sweep_spacing: float = 8.0):
        self.center = center.copy()
        self.width = width
        self.height = height
        self.altitude = altitude
        self.sweep_spacing = sweep_spacing
        self.status = MissionStatus.PENDING
        self.assigned_drones = []
        self.sectors: List[List[np.ndarray]] = []  # list of waypoint lists per sector
        self.drone_assignments: Dict[str, int] = {}  # drone_id -> sector index
        self.sectors_complete: List[bool] = []
        self.start_time = 0.0
        self.completion_pct = 0.0

    def plan(self, drones: Dict[str, DroneAgent]):
        self.status = MissionStatus.ACTIVE
        n = len(self.assigned_drones)
        if n == 0:
            return

        # Divide area into vertical strips, one per drone
        half_w = self.width / 2
        half_h = self.height / 2
        strip_width = self.width / n

        self.sectors = []
        for i in range(n):
            waypoints = []
            x_start = self.center[0] - half_w + i * strip_width
            x_end = x_start + strip_width
            x_mid = (x_start + x_end) / 2

            # Lawnmower pattern within strip
            y = self.center[1] - half_h
            direction = 1
            while y <= self.center[1] + half_h:
                if direction == 1:
                    waypoints.append(np.array([x_start + 1, y, self.altitude]))
                    waypoints.append(np.array([x_end - 1, y, self.altitude]))
                else:
                    waypoints.append(np.array([x_end - 1, y, self.altitude]))
                    waypoints.append(np.array([x_start + 1, y, self.altitude]))
                y += self.sweep_spacing
                direction *= -1
            self.sectors.append(waypoints)

        self.sectors_complete = [False] * n
        self.drone_assignments = {}

        for i, did in enumerate(self.assigned_drones):
            if did in drones and i < len(self.sectors):
                self.drone_assignments[did] = i
                drones[did].follow_waypoints(self.sectors[i])

    def tick(self, drones: Dict[str, DroneAgent], dt: float, sim_time: float):
        if self.status != MissionStatus.ACTIVE:
            return

        # Check drone progress — sector done if drone reached last waypoint
        for did, sector_idx in list(self.drone_assignments.items()):
            if did not in drones:
                continue
            drone = drones[did]
            sector_wps = self.sectors[sector_idx]
            if not sector_wps:
                continue
            # Done if: waypoint index past end, OR drone is near the final waypoint
            last_wp = sector_wps[-1]
            if drone.waypoint_idx >= len(sector_wps) or dist(drone.position, last_wp) < drone.waypoint_radius * 2:
                if drone.waypoint_idx > len(sector_wps) // 2:  # at least halfway through
                    self.sectors_complete[sector_idx] = True

        # Redistribute orphaned sectors
        active_drones = [d for d in self.assigned_drones if d in drones]
        orphaned = []
        for i, complete in enumerate(self.sectors_complete):
            if not complete:
                assigned_drone = None
                for did, si in self.drone_assignments.items():
                    if si == i:
                        assigned_drone = did
                        break
                if assigned_drone is None or assigned_drone not in drones:
                    orphaned.append(i)

        if orphaned and active_drones:
            # Assign orphaned sectors round-robin to active drones
            for i, sector_idx in enumerate(orphaned):
                recipient = active_drones[i % len(active_drones)]
                self.drone_assignments[recipient] = sector_idx
                if recipient in drones:
                    drones[recipient].follow_waypoints(self.sectors[sector_idx])

        # Update completion
        done = sum(self.sectors_complete)
        self.completion_pct = done / max(len(self.sectors_complete), 1) * 100

        if all(self.sectors_complete):
            self.status = MissionStatus.COMPLETE

    def get_status(self):
        return {
            "mission": self.name,
            "status": self.status.value,
            "completion": f"{self.completion_pct:.0f}%",
            "sectors": f"{sum(self.sectors_complete)}/{len(self.sectors_complete)}",
            "drones_active": len(self.drone_assignments),
        }


# ═══════════════════════════════════════════════════════════════════
# Perimeter Patrol
# ═══════════════════════════════════════════════════════════════════

class PerimeterPatrolMission(Mission):
    """
    Drones continuously orbit a perimeter at even spacing.
    Self-heals: if a drone drops, others spread out to cover the gap.
    """

    name = "perimeter_patrol"

    def __init__(self, center: np.ndarray, radius: float = 30.0,
                 altitude: float = 10.0, orbit_speed: float = 0.3):
        self.center = center.copy()
        self.radius = radius
        self.altitude = altitude
        self.orbit_speed = orbit_speed  # rad/s
        self.status = MissionStatus.PENDING
        self.assigned_drones = []
        self.drone_angles: Dict[str, float] = {}  # current angle per drone
        self.start_time = 0.0
        self.completion_pct = 0.0
        self.laps_completed: Dict[str, int] = {}

    def plan(self, drones: Dict[str, DroneAgent]):
        self.status = MissionStatus.ACTIVE
        n = len(self.assigned_drones)
        for i, did in enumerate(self.assigned_drones):
            angle = 2 * np.pi * i / n
            self.drone_angles[did] = angle
            self.laps_completed[did] = 0

    def tick(self, drones: Dict[str, DroneAgent], dt: float, sim_time: float):
        if self.status != MissionStatus.ACTIVE:
            return

        active = [d for d in self.assigned_drones if d in drones]
        n = len(active)
        if n == 0:
            self.status = MissionStatus.FAILED
            return

        # Re-space drones evenly if count changed
        target_spacing = 2 * np.pi / n

        for i, did in enumerate(active):
            # Advance angle
            target_angle = i * target_spacing + self.orbit_speed * sim_time
            self.drone_angles[did] = target_angle

            # Compute orbital position
            angle = target_angle
            target_pos = self.center.copy()
            target_pos[0] += self.radius * np.cos(angle)
            target_pos[1] += self.radius * np.sin(angle)
            target_pos[2] = self.altitude

            drones[did].target = target_pos
            if drones[did].mode not in (DroneMode.LANDING, DroneMode.IDLE, DroneMode.EMERGENCY):
                drones[did].mode = DroneMode.FORMATION

            # Track laps
            if angle > 2 * np.pi * (self.laps_completed.get(did, 0) + 1):
                self.laps_completed[did] = self.laps_completed.get(did, 0) + 1

        total_laps = sum(self.laps_completed.get(d, 0) for d in active)
        self.completion_pct = min(100, total_laps * 10)  # arbitrary progress metric

    def get_status(self):
        return {
            "mission": self.name,
            "status": self.status.value,
            "radius": self.radius,
            "drones_active": len([d for d in self.assigned_drones if d in (self.drone_angles or {})]),
            "total_laps": sum(self.laps_completed.values()),
        }


# ═══════════════════════════════════════════════════════════════════
# Target Tracking
# ═══════════════════════════════════════════════════════════════════

class TargetTrackMission(Mission):
    """
    Surround and follow a moving target.
    Drones form a dynamic ring that adjusts to target movement.
    """

    name = "target_track"

    def __init__(self, target_pos: np.ndarray, target_vel: np.ndarray = None,
                 standoff_radius: float = 15.0, altitude: float = 10.0):
        self.target_pos = target_pos.astype(float).copy()
        self.target_vel = target_vel.astype(float).copy() if target_vel is not None else np.zeros(3)
        self.standoff = standoff_radius
        self.altitude = altitude
        self.status = MissionStatus.PENDING
        self.assigned_drones = []
        self.start_time = 0.0
        self.completion_pct = 0.0
        self.track_time: float = 0.0

    def plan(self, drones: Dict[str, DroneAgent]):
        self.status = MissionStatus.ACTIVE

    def tick(self, drones: Dict[str, DroneAgent], dt: float, sim_time: float):
        if self.status != MissionStatus.ACTIVE:
            return

        # Move target
        self.target_pos += self.target_vel * dt

        active = [d for d in self.assigned_drones if d in drones]
        n = len(active)
        if n == 0:
            self.status = MissionStatus.FAILED
            return

        # Distribute drones in a ring around target
        for i, did in enumerate(active):
            angle = 2 * np.pi * i / n + sim_time * 0.1  # slow rotation
            pos = self.target_pos.copy()
            pos[0] += self.standoff * np.cos(angle)
            pos[1] += self.standoff * np.sin(angle)
            pos[2] = self.altitude

            drones[did].target = pos
            if drones[did].mode not in (DroneMode.LANDING, DroneMode.IDLE, DroneMode.EMERGENCY):
                drones[did].mode = DroneMode.FORMATION

        self.track_time += dt
        self.completion_pct = min(100, self.track_time * 2)

    def update_target(self, pos: np.ndarray, vel: np.ndarray = None):
        """Update target position externally (e.g., from sensor data)."""
        self.target_pos = pos.astype(float).copy()
        if vel is not None:
            self.target_vel = vel.astype(float).copy()

    def get_status(self):
        return {
            "mission": self.name,
            "status": self.status.value,
            "target_pos": self.target_pos.tolist(),
            "track_time": f"{self.track_time:.1f}s",
            "drones_active": len(self.assigned_drones),
        }


# ═══════════════════════════════════════════════════════════════════
# Relay Chain
# ═══════════════════════════════════════════════════════════════════

class RelayChainMission(Mission):
    """
    Form a communication relay chain between two points.
    Drones space themselves evenly to maintain connectivity.
    """

    name = "relay_chain"

    def __init__(self, point_a: np.ndarray, point_b: np.ndarray, altitude: float = 15.0):
        self.point_a = point_a.astype(float).copy()
        self.point_b = point_b.astype(float).copy()
        self.altitude = altitude
        self.status = MissionStatus.PENDING
        self.assigned_drones = []
        self.start_time = 0.0
        self.completion_pct = 0.0
        self.chain_established = False

    def plan(self, drones: Dict[str, DroneAgent]):
        self.status = MissionStatus.ACTIVE

    def tick(self, drones: Dict[str, DroneAgent], dt: float, sim_time: float):
        if self.status != MissionStatus.ACTIVE:
            return

        active = [d for d in self.assigned_drones if d in drones]
        n = len(active)
        if n == 0:
            self.status = MissionStatus.FAILED
            return

        # Space drones evenly between A and B
        for i, did in enumerate(active):
            t = (i + 1) / (n + 1)  # evenly spaced 0..1
            pos = self.point_a + t * (self.point_b - self.point_a)
            pos[2] = self.altitude

            drones[did].target = pos
            if drones[did].mode not in (DroneMode.LANDING, DroneMode.IDLE, DroneMode.EMERGENCY):
                drones[did].mode = DroneMode.FORMATION

        # Check if chain is established (all drones near their positions)
        all_in_place = True
        for did in active:
            if did in drones and drones[did].target is not None:
                if dist(drones[did].position, drones[did].target) > 5.0:
                    all_in_place = False
                    break

        self.chain_established = all_in_place
        self.completion_pct = 100.0 if all_in_place else 50.0

    def get_status(self):
        return {
            "mission": self.name,
            "status": self.status.value,
            "chain_established": self.chain_established,
            "drones_active": len(self.assigned_drones),
            "distance": f"{dist(self.point_a, self.point_b):.0f}m",
        }


# ═══════════════════════════════════════════════════════════════════
# Mission Planner
# ═══════════════════════════════════════════════════════════════════

class MissionPlanner:
    """
    Manages active missions and assigns drones to them.

    Usage:
        planner = MissionPlanner()
        mission = AreaSearchMission(center, width=100, height=100)
        planner.assign_mission(mission, ["drone_000", "drone_001", ...])

        # each tick:
        planner.tick(drones, dt, sim_time)
    """

    def __init__(self):
        self.missions: List[Mission] = []
        self.mission_history: List[Dict] = []

    def assign_mission(self, mission: Mission, drone_ids: List[str],
                       drones: Dict[str, DroneAgent] = None):
        """Assign a mission to a set of drones and begin planning."""
        mission.assigned_drones = list(drone_ids)
        if drones:
            mission.plan(drones)
        self.missions.append(mission)

    def tick(self, drones: Dict[str, DroneAgent], dt: float, sim_time: float):
        """Update all active missions."""
        for mission in self.missions:
            if mission.status == MissionStatus.ACTIVE:
                mission.tick(drones, dt, sim_time)

            # Log completed/failed
            if mission.status in (MissionStatus.COMPLETE, MissionStatus.FAILED):
                if mission not in [m.get("mission") for m in self.mission_history]:
                    self.mission_history.append({
                        "mission": mission,
                        "name": mission.name,
                        "status": mission.status.value,
                        "completion": mission.completion_pct,
                    })

    def get_active_missions(self) -> List[Mission]:
        return [m for m in self.missions if m.status == MissionStatus.ACTIVE]

    def cancel_all(self):
        for m in self.missions:
            if m.status == MissionStatus.ACTIVE:
                m.status = MissionStatus.FAILED

    def status(self) -> str:
        active = len(self.get_active_missions())
        total = len(self.missions)
        return f"Missions: {active} active / {total} total"
