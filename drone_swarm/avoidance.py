"""
avoidance.py — Obstacle Avoidance & Pathfinding
Three-layer real-time avoidance (reactive + predictive + cooperative)
plus A* and potential-field planners for route planning.
"""

import numpy as np
import heapq
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from abc import ABC, abstractmethod

from drone import normalize, clamp_mag, dist


# =============================================================================
# Real-Time Obstacle Avoidance (runs every tick)
# =============================================================================

@dataclass
class AvoidanceConfig:
    # Reactive (potential field repulsion)
    repulsion_radius: float = 15.0
    hard_radius: float = 3.0
    repulsion_strength: float = 10.0
    # Predictive (velocity obstacle look-ahead)
    lookahead_time: float = 2.0
    predictive_strength: float = 5.0
    # Cooperative (drone-drone separation)
    min_separation: float = 4.0
    separation_strength: float = 8.0
    # Global cap
    max_avoidance_accel: float = 10.0


class ObstacleAvoidanceSystem:
    """
    Multi-layered avoidance. Call compute() each tick per drone.

    Layer 1 — Reactive:    Potential field repulsion from nearby obstacles.
    Layer 2 — Predictive:  Ray-cast along velocity to detect future collisions.
    Layer 3 — Cooperative: Repulsion between drones to prevent mid-air collisions.

    Subclass and override any layer to customize:
        class MyAvoidance(ObstacleAvoidanceSystem):
            def reactive(self, drone, obstacles):
                ...  # your logic
    """

    def __init__(self, config: AvoidanceConfig = None):
        self.config = config or AvoidanceConfig()

    def compute(self, drone, obstacles: List[Dict], all_drones: Dict) -> np.ndarray:
        total = (self.reactive(drone, obstacles)
                 + self.predictive(drone, obstacles)
                 + self.cooperative(drone, all_drones))
        return clamp_mag(total, self.config.max_avoidance_accel)

    def reactive(self, drone, obstacles: List[Dict]) -> np.ndarray:
        """Inverse-distance repulsion. Hard zone triggers emergency push."""
        cfg = self.config
        steer = np.zeros(3)
        for obs in obstacles:
            d = dist(drone.position, obs["position"]) - obs.get("radius", 1.0)
            if d < cfg.repulsion_radius and d > 0.01:
                away = normalize(drone.position - obs["position"])
                if d < cfg.hard_radius:
                    force = cfg.repulsion_strength * (cfg.hard_radius / d) ** 2
                else:
                    ratio = 1.0 - (d - cfg.hard_radius) / (cfg.repulsion_radius - cfg.hard_radius)
                    force = cfg.repulsion_strength * ratio
                steer += away * force
            elif d <= 0.01:
                steer += normalize(np.random.randn(3)) * cfg.repulsion_strength * 5
        return steer

    def predictive(self, drone, obstacles: List[Dict]) -> np.ndarray:
        """Look ahead along velocity; steer perpendicular if collision predicted."""
        cfg = self.config
        steer = np.zeros(3)
        vel = drone.velocity
        speed = np.linalg.norm(vel)
        if speed < 0.5:
            return steer
        direction = normalize(vel)

        for obs in obstacles:
            obs_vel = obs.get("velocity", np.zeros(3))
            rel_vel = vel - obs_vel
            expanded_r = obs.get("radius", 1.0) + 1.5

            # Ray-sphere intersection
            oc = drone.position - obs["position"]
            d_dir = normalize(rel_vel)
            b = 2.0 * np.dot(oc, d_dir)
            c = np.dot(oc, oc) - expanded_r ** 2
            disc = b * b - 4.0 * c
            if disc < 0:
                continue
            t_hit = (-b - np.sqrt(disc)) / 2.0
            if t_hit < 0 or t_hit > cfg.lookahead_time:
                continue

            # Steer perpendicular to avoid
            collision_pt = drone.position + rel_vel * t_hit
            perp = np.cross(direction, normalize(collision_pt - obs["position"]))
            if np.linalg.norm(perp) < 0.01:
                perp = np.cross(direction, np.array([0, 0, 1]))
            urgency = 1.0 - (t_hit / cfg.lookahead_time)
            steer += normalize(perp) * cfg.predictive_strength * urgency

        return steer

    def cooperative(self, drone, all_drones: Dict) -> np.ndarray:
        """Maintain minimum separation between drones."""
        cfg = self.config
        steer = np.zeros(3)
        for other in all_drones.values():
            if other.id == drone.id:
                continue
            d = dist(drone.position, other.position)
            if 0 < d < cfg.min_separation:
                away = normalize(drone.position - other.position)
                steer += away * cfg.separation_strength * (1.0 - d / cfg.min_separation) ** 2
        return steer


# =============================================================================
# Pathfinding (for route planning, not per-tick)
# =============================================================================

class PathPlanner(ABC):
    """Base class. Subclass to add your own planner."""
    @abstractmethod
    def plan(self, start: np.ndarray, goal: np.ndarray,
             obstacles: List[Dict], bounds=None) -> List[np.ndarray]:
        pass


class AStarPlanner(PathPlanner):
    """
    A* on a discretized 3D grid with path smoothing.
    Best for: structured environments, optimal paths.

    Usage:
        planner = AStarPlanner(resolution=2.0)
        path = planner.plan(start, goal, obstacles)
        drone.follow_waypoints(path)
    """

    def __init__(self, resolution: float = 2.0, safety_margin: float = 2.0,
                 diagonal: bool = True, max_iter: int = 10000):
        self.res = resolution
        self.margin = safety_margin
        self.diagonal = diagonal
        self.max_iter = max_iter

    def plan(self, start, goal, obstacles, bounds=None):
        if bounds is None:
            pts = [start, goal] + [o["position"] for o in obstacles]
            mins = np.min(pts, axis=0) - 20
            maxs = np.max(pts, axis=0) + 20
        else:
            mins, maxs = bounds

        to_grid = lambda p: tuple(((p - mins) / self.res).astype(int))
        to_world = lambda g: np.array(g) * self.res + mins

        def valid(gp):
            wp = to_world(gp)
            if np.any(wp < mins) or np.any(wp > maxs):
                return False
            for obs in obstacles:
                if dist(wp, obs["position"]) < obs.get("radius", 1.0) + self.margin:
                    return False
            return True

        # 26-connected or 6-connected neighbors
        if self.diagonal:
            offsets = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)
                       if not (dx==0 and dy==0 and dz==0)]
        else:
            offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

        start_g, goal_g = to_grid(start), to_grid(goal)
        open_set = [(0.0, start_g)]
        came_from = {}
        g_score = {start_g: 0.0}
        closed: Set = set()

        for _ in range(self.max_iter):
            if not open_set:
                break
            _, current = heapq.heappop(open_set)
            if current == goal_g:
                path = [goal.copy()]
                node = goal_g
                while node in came_from:
                    node = came_from[node]
                    path.append(to_world(node))
                path.reverse()
                path[0] = start.copy()
                return self._smooth(path, obstacles)
            if current in closed:
                continue
            closed.add(current)
            for off in offsets:
                nb = tuple(c + o for c, o in zip(current, off))
                if nb in closed or not valid(nb):
                    continue
                cost = g_score[current] + dist(np.array(current, float), np.array(nb, float))
                if cost < g_score.get(nb, float('inf')):
                    came_from[nb] = current
                    g_score[nb] = cost
                    h = dist(np.array(nb, float), np.array(goal_g, float))
                    heapq.heappush(open_set, (cost + h, nb))

        return [start.copy(), goal.copy()]

    def _smooth(self, path, obstacles):
        if len(path) <= 2:
            return path
        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            farthest = i + 1
            for j in range(len(path) - 1, i, -1):
                if self._line_clear(result[-1], path[j], obstacles):
                    farthest = j
                    break
            result.append(path[farthest])
            i = farthest
        return result

    def _line_clear(self, a, b, obstacles):
        steps = max(int(dist(a, b) / (self.res * 0.5)), 2)
        for t in np.linspace(0, 1, steps):
            pt = a + t * (b - a)
            for obs in obstacles:
                if dist(pt, obs["position"]) < obs.get("radius", 1.0) + self.margin:
                    return False
        return True


class PotentialFieldPlanner(PathPlanner):
    """
    Gradient descent on attractive/repulsive fields.
    Fast but can get stuck in local minima. Good for open environments.
    """

    def __init__(self, attract: float = 1.0, repulse: float = 50.0,
                 repulse_range: float = 10.0, step: float = 1.0, max_steps: int = 500):
        self.attract = attract
        self.repulse = repulse
        self.repulse_range = repulse_range
        self.step = step
        self.max_steps = max_steps

    def plan(self, start, goal, obstacles, bounds=None):
        path = [start.copy()]
        pos = start.copy()
        for _ in range(self.max_steps):
            if dist(pos, goal) < 2.0:
                path.append(goal.copy())
                return path
            f_att = self.attract * normalize(goal - pos)
            f_rep = np.zeros(3)
            for obs in obstacles:
                d = dist(pos, obs["position"]) - obs.get("radius", 1.0)
                if 0 < d < self.repulse_range:
                    away = normalize(pos - obs["position"])
                    f_rep += self.repulse * (1/d - 1/self.repulse_range) * (1/d**2) * away
            pos = pos + normalize(f_att + f_rep) * self.step
            path.append(pos.copy())
        path.append(goal.copy())
        return path
