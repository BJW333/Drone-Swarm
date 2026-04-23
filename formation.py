"""
formation.py — Formation Control & Swarm Intelligence
Configurable formations (V, circle, grid, etc.) plus
Boids flocking and Particle Swarm Optimization.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any
from abc import ABC, abstractmethod

from drone import normalize, clamp_mag, dist


#Formation Control

class FormationType(Enum):
    NONE = "none"
    LINE = "line"
    V_SHAPE = "v_shape"
    CIRCLE = "circle"
    GRID = "grid"
    DIAMOND = "diamond"
    SPIRAL = "spiral"
    CUSTOM = "custom"


class FormationStrategy(ABC):
    """Subclass to define your own formation geometry."""
    @abstractmethod
    def compute_positions(self, center: np.ndarray, n: int, **params) -> List[np.ndarray]:
        pass


def _rotate_z(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c, v[2]])


class FormationController:
    """
    Computes target positions for a formation.

    Usage:
        ctrl = FormationController()
        targets = ctrl.compute(FormationType.V_SHAPE, 10, center)
        for drone, target in zip(drones, targets):
            drone.target = target

    Custom formations:
        class Arrow(FormationStrategy):
            def compute_positions(self, center, n, **p):
                return [...]
        ctrl.register("arrow", Arrow())
        ctrl.compute(FormationType.CUSTOM, 10, center, name="arrow")
    """

    def __init__(self):
        self._custom: Dict[str, FormationStrategy] = {}

    def register(self, name: str, strategy: FormationStrategy):
        self._custom[name] = strategy

    def compute(self, formation: FormationType, n: int,
                center: np.ndarray, **params) -> List[np.ndarray]:
        builders = {
            FormationType.LINE: self._line,
            FormationType.V_SHAPE: self._v_shape,
            FormationType.CIRCLE: self._circle,
            FormationType.GRID: self._grid,
            FormationType.DIAMOND: self._diamond,
            FormationType.SPIRAL: self._spiral,
        }
        if formation == FormationType.CUSTOM:
            name = params.get("name", "")
            if name in self._custom:
                return self._custom[name].compute_positions(center, n, **params)
        builder = builders.get(formation)
        if builder:
            return builder(center, n, **params)
        return [center.copy() for _ in range(n)]

    def _line(self, center, n, spacing=4.0, heading=0.0, **_):
        return [center + _rotate_z(np.array([(i-(n-1)/2)*spacing, 0, 0]), heading)
                for i in range(n)]

    def _v_shape(self, center, n, spacing=5.0, angle_deg=30, heading=0.0, **_):
        angle = np.radians(angle_deg)
        positions = [center.copy()]
        for i in range(1, n):
            side = 1 if i % 2 else -1
            rank = (i + 1) // 2
            offset = _rotate_z(np.array([
                -rank * spacing * np.cos(angle),
                side * rank * spacing * np.sin(angle), 0
            ]), heading)
            positions.append(center + offset)
        return positions

    def _circle(self, center, n, radius=10.0, **_):
        return [center + np.array([radius*np.cos(2*np.pi*i/n),
                                   radius*np.sin(2*np.pi*i/n), 0])
                for i in range(n)]

    def _grid(self, center, n, spacing=4.0, **_):
        cols = int(np.ceil(np.sqrt(n)))
        return [center + np.array([(i%cols - (cols-1)/2)*spacing,
                                   (i//cols - ((n//cols)-1)/2)*spacing, 0])
                for i in range(n)]

    def _diamond(self, center, n, spacing=5.0, **_):
        positions = [center.copy()]
        layer, placed = 1, 1
        while placed < n:
            for side in range(4):
                for j in range(layer):
                    if placed >= n:
                        break
                    angle = side * np.pi/2 + (j/layer)*(np.pi/2)
                    positions.append(center + np.array([
                        layer*spacing*np.cos(angle),
                        layer*spacing*np.sin(angle), 0]))
                    placed += 1
            layer += 1
        return positions[:n]

    def _spiral(self, center, n, spacing=3.0, height_step=1.0, **_):
        golden = (1 + np.sqrt(5)) / 2
        positions = []
        for i in range(n):
            angle = 2 * np.pi * i / golden
            r = spacing * np.sqrt(i + 1)
            positions.append(center + np.array([
                r*np.cos(angle), r*np.sin(angle), i*height_step]))
        return positions


#Boids Flocking 

@dataclass
class BoidsConfig:
    separation_radius: float = 6.0
    alignment_radius: float = 15.0
    cohesion_radius: float = 20.0
    separation_weight: float = 2.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 0.8
    max_force: float = 5.0


class BoidsBehavior:
    """
    Reynolds Boids: separation + alignment + cohesion.
    Returns steering acceleration per drone.

    Usage:
        boids = BoidsBehavior()
        vectors = boids.compute_all(drone_list)
        for drone, vec in zip(drone_list, vectors):
            drone.add_plugin_vector("separation", vec)
    """

    def __init__(self, config: BoidsConfig = None):
        self.config = config or BoidsConfig()

    def compute_all(self, drones: list) -> List[np.ndarray]:
        positions = [d.position for d in drones]
        velocities = [d.velocity for d in drones]
        return [self._compute_one(i, positions, velocities) for i in range(len(drones))]

    def _compute_one(self, idx, positions, velocities):
        cfg = self.config
        pos, vel = positions[idx], velocities[idx]
        sep = ali = coh = np.zeros(3)
        sc = ac = cc = 0

        for j, (op, ov) in enumerate(zip(positions, velocities)):
            if j == idx:
                continue
            d = dist(pos, op)
            if 0 < d < cfg.separation_radius:
                sep = sep + normalize(pos - op) / (d + 0.01)
                sc += 1
            if d < cfg.alignment_radius:
                ali = ali + ov
                ac += 1
            if d < cfg.cohesion_radius:
                coh = coh + op
                cc += 1

        steer = np.zeros(3)
        if sc > 0:
            steer += cfg.separation_weight * sep / sc
        if ac > 0:
            steer += cfg.alignment_weight * normalize(ali/ac - vel)
        if cc > 0:
            steer += cfg.cohesion_weight * normalize(coh/cc - pos)
        return clamp_mag(steer, cfg.max_force)



#Particle Swarm Optimization (cooperative search algorithm)

class ParticleSwarmOptimizer:
    """
    PSO for distributed search tasks (e.g., find a signal source, map an area).
    Each drone is a particle evaluating a fitness function.

    Usage:
        pso = ParticleSwarmOptimizer()
        def fitness(pos): return -dist(pos, target)
        adjustments = pso.step(drones, fitness)
    """

    def __init__(self, inertia=0.7, cognitive=1.5, social=1.5, max_vel=5.0):
        self.w = inertia
        self.c1 = cognitive
        self.c2 = social
        self.max_vel = max_vel
        self._pbest: Dict[str, tuple] = {}
        self._gbest: tuple = None

    def step(self, drones, fitness_fn) -> List[np.ndarray]:
        for d in drones:
            f = fitness_fn(d.position)
            pb = self._pbest.get(d.id)
            if pb is None or f > pb[1]:
                self._pbest[d.id] = (d.position.copy(), f)
            if self._gbest is None or f > self._gbest[1]:
                self._gbest = (d.position.copy(), f)

        results = []
        for d in drones:
            r1, r2 = np.random.random(3), np.random.random(3)
            cognitive = self.c1 * r1 * (self._pbest[d.id][0] - d.position)
            social = self.c2 * r2 * (self._gbest[0] - d.position)
            new_vel = self.w * d.velocity + cognitive + social
            results.append(clamp_mag(new_vel, self.max_vel))
        return results

    def reset(self):
        self._pbest.clear()
        self._gbest = None
