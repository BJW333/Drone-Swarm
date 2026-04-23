#!/usr/bin/env python3
"""
visualize.py — Live 3D Drone Swarm Visualization
Opens an animated 3D matplotlib window with X/Y/Z grid box.
Click and drag to rotate the view. Scroll to zoom.

Drop next to your framework files and run:
    python3.10 visualize.py
    python3.10 visualize.py --drones 20
    python3.10 visualize.py --scenario formation
    python3.10 visualize.py --scenario dense
    python3.10 visualize.py --scenario moving
    python3.10 visualize.py --scenario swarm_nav
    python3.10 visualize.py --speed 2

Controls:
    Mouse drag — rotate view
    Scroll     — zoom
    SPACE      — pause / resume
    Q          — quit
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from drone import DroneMode, PhysicsConfig
from formation import FormationType
from swarm import SwarmManager, SwarmConfig, GeofencePlugin


# ═══════════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════════

def build_obstacle_run(n=12):
    swarm = SwarmManager(SwarmConfig(num_drones=n, tick_rate=30, default_altitude=10))
    swarm.add_obstacle(np.array([25.0, 0.0, 10.0]),   5.0)
    swarm.add_obstacle(np.array([-10.0, 18.0, 10.0]), 4.0)
    swarm.add_obstacle(np.array([5.0, -22.0, 10.0]),  6.0)
    swarm.add_obstacle(np.array([35.0, 22.0, 10.0]),  4.5)
    swarm.add_obstacle(np.array([-20.0, -8.0, 10.0]), 3.5)
    swarm.plugins.register(GeofencePlugin())
    swarm.takeoff_all()
    def on_tick(s):
        if abs(s.elapsed_time - 2.5) < s.dt:
            s.set_formation(FormationType.V_SHAPE, center=np.array([42.0, 32.0, 10.0]))
    swarm.on_tick(on_tick)
    return swarm, "Obstacle Run"


def build_formation_cycle(n=15):
    swarm = SwarmManager(SwarmConfig(num_drones=n, tick_rate=30, default_altitude=10))
    swarm.plugins.register(GeofencePlugin())
    swarm.takeoff_all()
    schedule = [
        (2.5,  FormationType.CIRCLE,  {"radius": 20.0}),
        (7.0,  FormationType.V_SHAPE, {}),
        (11.5, FormationType.GRID,    {}),
        (16.0, FormationType.LINE,    {"spacing": 5.0}),
        (20.5, FormationType.DIAMOND, {}),
        (25.0, FormationType.SPIRAL,  {}),
        (29.5, FormationType.CIRCLE,  {"radius": 18.0}),
    ]
    def on_tick(s):
        for t, fmt, p in schedule:
            if abs(s.elapsed_time - t) < s.dt:
                s.set_formation(fmt, center=np.array([0.0, 0.0, 10.0]), **p)
    swarm.on_tick(on_tick)
    return swarm, "Formation Cycle"


def build_dense_field(n=12):
    swarm = SwarmManager(SwarmConfig(num_drones=n, tick_rate=30, default_altitude=10))
    rng = np.random.RandomState(42)
    for _ in range(18):
        pos = np.array([rng.uniform(-35, 35), rng.uniform(-35, 35), rng.uniform(5, 20)])
        swarm.add_obstacle(pos, rng.uniform(2.0, 5.0))
    swarm.plugins.register(GeofencePlugin())
    swarm.takeoff_all()
    def on_tick(s):
        if abs(s.elapsed_time - 3.0) < s.dt:
            s.send_all_to(np.array([32.0, 28.0, 12.0]))
    swarm.on_tick(on_tick)
    return swarm, "Dense Field (18 obstacles)"


def build_moving_dodge(n=12):
    swarm = SwarmManager(SwarmConfig(num_drones=n, tick_rate=30, default_altitude=10))
    swarm.add_obstacle(np.array([-45.0, 0.0, 10.0]),  4.0, np.array([3.5, 0.5, 0.0]))
    swarm.add_obstacle(np.array([0.0, -45.0, 10.0]),  5.0, np.array([0.5, 3.0, 0.0]))
    swarm.add_obstacle(np.array([40.0, 35.0, 10.0]),  3.5, np.array([-2.5, -2.0, 0.0]))
    swarm.add_obstacle(np.array([35.0, -30.0, 10.0]), 4.5, np.array([-2.0, 2.5, 0.0]))
    swarm.plugins.register(GeofencePlugin())
    swarm.takeoff_all()
    def on_tick(s):
        if abs(s.elapsed_time - 2.0) < s.dt:
            s.set_formation(FormationType.CIRCLE, center=np.array([0.0, 0.0, 10.0]), radius=16.0)
    swarm.on_tick(on_tick)
    return swarm, "Moving Obstacles"


def build_swarm_navigate(n=12):
    swarm = SwarmManager(SwarmConfig(num_drones=n, tick_rate=30, default_altitude=10))
    swarm.add_obstacle(np.array([0.0, 20.0, 10.0]),    4.0)
    swarm.add_obstacle(np.array([20.0, -10.0, 10.0]),  5.0)
    swarm.add_obstacle(np.array([-15.0, -20.0, 10.0]), 3.5)
    swarm.plugins.register(GeofencePlugin())
    swarm.takeoff_all()
    waypoints = [
        (3.0,  np.array([-30.0, 25.0, 10.0])),
        (8.0,  np.array([30.0, -25.0, 10.0])),
        (13.0, np.array([-25.0, -30.0, 10.0])),
        (18.0, np.array([25.0, 30.0, 10.0])),
        (23.0, np.array([0.0, 0.0, 10.0])),
    ]
    def on_tick(s):
        for t, wp in waypoints:
            if abs(s.elapsed_time - t) < s.dt:
                s.set_formation(FormationType.CIRCLE, center=wp, radius=12.0)
    swarm.on_tick(on_tick)
    return swarm, "Swarm Navigate"


SCENARIOS = {
    "obstacle":  build_obstacle_run,
    "formation": build_formation_cycle,
    "dense":     build_dense_field,
    "moving":    build_moving_dodge,
    "swarm_nav": build_swarm_navigate,
}


# ═══════════════════════════════════════════════════════════════════
# Obstacle Wireframe
# ═══════════════════════════════════════════════════════════════════

def draw_obstacle(ax, pos, radius):
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 7)
    x = pos[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = pos[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = pos[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#CC3333", alpha=0.35, linewidth=0.5)


# ═══════════════════════════════════════════════════════════════════
# Visualizer
# ═══════════════════════════════════════════════════════════════════

class Visualizer3D:
    def __init__(self, scenario="obstacle", n_drones=12, speed=1.0, bounds=60):
        self.bounds = bounds
        self.speed = speed
        builder = SCENARIOS[scenario]
        self.swarm, self.label = builder(n_drones)

        self.trails = {did: [] for did in self.swarm.drones}
        self.trail_len = 50

        n = len(self.swarm.drones)
        self.drone_colors = []
        for i in range(n):
            val = 0.55 + 0.35 * (i / max(n - 1, 1))
            self.drone_colors.append((val * 0.85, val * 0.9, val, 0.9))

        self.paused = False
        self._setup()

    def _setup(self):
        self.fig = plt.figure(figsize=(12, 9), facecolor="white")
        self.fig.canvas.manager.set_window_title("DroneSwarm — 3D Visualization")
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.ax = self.fig.add_subplot(111, projection="3d")
        # Set the starting angle ONCE — after this, mouse drag controls it
        self.ax.view_init(elev=28, azim=-55)

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "q":
            plt.close(self.fig)

    def _update(self, frame):
        if not self.paused:
            steps = max(1, int(self.speed))
            for _ in range(steps):
                self.swarm.tick()

        # Save current camera angle before clearing
        elev = self.ax.elev
        azim = self.ax.azim

        ax = self.ax
        ax.clear()

        # Restore camera angle after clear
        ax.view_init(elev=elev, azim=azim)

        B = self.bounds
        ax.set_xlim(-B, B)
        ax.set_ylim(-B, B)
        ax.set_zlim(0, B * 0.8)
        ax.set_xlabel("X (m)", fontsize=8, labelpad=6)
        ax.set_ylabel("Y (m)", fontsize=8, labelpad=6)
        ax.set_zlabel("Z (m)", fontsize=8, labelpad=6)
        ax.tick_params(labelsize=6, pad=2)

        # ── Ground grid ──
        grid = np.linspace(-B, B, 13)
        for g in grid:
            ax.plot([g, g], [-B, B], [0, 0], color="#CCCCCC", linewidth=0.3, alpha=0.5)
            ax.plot([-B, B], [g, g], [0, 0], color="#CCCCCC", linewidth=0.3, alpha=0.5)

        # ── Obstacles ──
        for obs in self.swarm.obstacles:
            draw_obstacle(ax, obs.position, obs.radius)

        # ── Drones ──
        for i, (did, d) in enumerate(self.swarm.drones.items()):
            color = self.drone_colors[i]
            pos = d.position
            vel = d.velocity

            self.trails.setdefault(did, [])
            self.trails[did].append(pos.copy())
            if len(self.trails[did]) > self.trail_len:
                self.trails[did] = self.trails[did][-self.trail_len:]

            # Drone marker
            ax.scatter(pos[0], pos[1], pos[2],
                       color=color, s=45, marker="^",
                       edgecolors="#444444", linewidths=0.3, depthshade=True, zorder=5)

            # Velocity quiver
            spd = np.linalg.norm(vel)
            if spd > 0.3:
                ax.quiver(pos[0], pos[1], pos[2],
                          vel[0]*0.4, vel[1]*0.4, vel[2]*0.4,
                          color="#555555", alpha=0.4,
                          arrow_length_ratio=0.3, linewidth=0.6)

            # Trail
            trail = np.array(self.trails[did])
            if len(trail) > 1:
                ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                        color="#888888", alpha=0.15, linewidth=0.8)

            # Ground shadow
            ax.scatter(pos[0], pos[1], 0,
                       color="#AAAAAA", s=8, alpha=0.15, marker=".", zorder=1)

            # Target line
            if d.target is not None:
                t = d.target
                ax.plot([pos[0], t[0]], [pos[1], t[1]], [pos[2], t[2]],
                        color="#AAAAAA", alpha=0.08, linewidth=0.4, linestyle="--")

        # ── HUD ──
        modes = {}
        for d in self.swarm.drones.values():
            modes[d.mode.value] = modes.get(d.mode.value, 0) + 1
        mode_str = "  ".join(f"{m}:{c}" for m, c in modes.items())
        status = "PAUSED" if self.paused else "RUNNING"

        ax.set_title(
            f"{self.label}  |  {len(self.swarm.drones)} drones  "
            f"{len(self.swarm.obstacles)} obstacles  |  "
            f"t={self.swarm.elapsed_time:.1f}s  {mode_str}  [{status}]\n"
            f"Drag to rotate  |  SPACE: pause  |  Q: quit",
            fontsize=8, fontfamily="monospace", color="#444444", pad=12
        )

    def run(self, duration=120):
        frames = int(duration * self.swarm.config.tick_rate / max(1, self.speed))
        self.anim = FuncAnimation(
            self.fig, self._update, frames=frames,
            interval=1000 / self.swarm.config.tick_rate,
            blit=False, repeat=True,
        )
        plt.show()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="3D Drone Swarm Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  obstacle   — V-formation through 5 static obstacles
  formation  — Cycle all 6 formations
  dense      — Navigate through 18 random obstacles
  moving     — Dodge 4 moving obstacles
  swarm_nav  — Hop between waypoints

Controls:
  Mouse drag   rotate view
  Scroll       zoom
  SPACE        pause / resume
  Q            quit
        """)
    parser.add_argument("--scenario", "-s", default="obstacle",
                        choices=list(SCENARIOS.keys()))
    parser.add_argument("--drones", "-n", type=int, default=12)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--bounds", type=int, default=60)
    args = parser.parse_args()

    print(f"  DroneSwarm 3D  |  {args.scenario}  |  {args.drones} drones  |  {args.speed}x")
    print(f"  Drag to rotate  |  SPACE=pause  |  Q=quit")

    viz = Visualizer3D(
        scenario=args.scenario,
        n_drones=args.drones,
        speed=args.speed,
        bounds=args.bounds,
    )
    viz.run(duration=args.duration)


if __name__ == "__main__":
    main()
