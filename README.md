# DroneSwarm AI

A modular Python framework for simulating autonomous drone swarms with decentralized intelligence. Supports multi-layer obstacle avoidance, mesh network communications, leader election, mission planning, and real-time 3D visualization.

---

## Features

**Drone Agents** — Each drone runs an independent sense → decide → act loop with PD control, waypoint navigation, and pluggable steering vectors for avoidance, formation, and custom behaviors.

**Obstacle Avoidance** — Three-layer real-time system: reactive potential field repulsion, predictive velocity-obstacle ray casting, and cooperative drone-drone separation. Includes A* and potential field path planners for route-level planning.

**Formation Control** — Seven built-in formations (V-shape, circle, grid, line, diamond, spiral, custom) plus Craig Reynolds boids flocking (separation, alignment, cohesion) and a particle swarm optimizer.

**Mesh Network Simulation** — Realistic comms model with distance-based packet loss, RSSI tracking, latency/jitter, bandwidth limits, message prioritization, multi-hop relay with TTL, heartbeat-based neighbor discovery, and network partition detection. Supports jamming simulation.

**Decentralized Consensus** — No central controller required. Drones elect leaders via a bully algorithm, vote on swarm health state (nominal → degraded → critical → partitioned), assign roles (leader, wingman, scout, relay), and fall back to autonomous operation when isolated.

**Mission Planning** — High-level objectives decomposed into per-drone tasks with automatic re-planning when drones are lost:
- **Area Search** — sector-based systematic sweep
- **Perimeter Patrol** — evenly-spaced orbital patrol
- **Target Track** — surround and follow a moving target
- **Relay Chain** — form a communication bridge between two points

**Performance Metrics** — Tracks formation accuracy, obstacle clearance margins, swarm cohesion/spread, comms delivery rate, mission completion, per-drone telemetry, and resilience. Exportable to JSON.

**Plugin System** — Extend behavior without modifying core code. Built-in plugins: geofence enforcement, telemetry logging, and battery simulation with auto-land.

**3D Visualization** — Interactive matplotlib window with drone markers, velocity vectors, trails, obstacle wireframes, and live status overlay. Drag to rotate, scroll to zoom.

---

## Architecture

```
main.py              CLI entry point, scenarios, mission setup
├── swarm.py         SwarmManager — tick loop, plugin system, obstacle management
│   ├── drone.py     DroneAgent — physics, sensing, PD controller, state machine
│   ├── avoidance.py Reactive + predictive + cooperative avoidance, A*, potential fields
│   ├── formation.py Formation strategies, boids flocking, PSO
│   ├── comms.py     MeshNetwork — lossy/delayed comms, RSSI, partitioning
│   ├── consensus.py Leader election, health voting, role assignment, autonomy
│   ├── missions.py  Mission base + area search, patrol, tracking, relay
│   └── metrics.py   MetricsCollector — per-tick performance snapshots
└── visualize.py     Standalone 3D visualization with scenario presets
```

---

## Quick Start

### Install

```bash
git clone https://github.com/BJW333/drone_swarm.git
cd drone_swarm
pip3.10 install -r requirements.txt
```

### Run

```bash
# Default: 10 drones, headless, 30 seconds
python3.10 main.py

# 20 drones with 3D visualization
python3.10 main.py --drones 20 --sim

# Start in V formation
python3.10 main.py --formation v_shape --sim

# Run a preset scenario
python3.10 main.py --scenario obstacle_run --sim
```

---

## Usage

### Missions

Assign drones to high-level objectives. Missions re-plan automatically if drones are lost.

```bash
python3.10 main.py --mission search         # area search sweep
python3.10 main.py --mission patrol         # perimeter patrol
python3.10 main.py --mission track          # target tracking
python3.10 main.py --mission relay          # comm relay chain
python3.10 main.py --mission all            # split drones across all missions
```

### Stress Testing

Simulate drone loss and communications jamming to test swarm resilience.

```bash
python3.10 main.py --kill 4                 # destroy 4 drones at t=10s
python3.10 main.py --jam                    # jam comms at t=15s
python3.10 main.py --mission all --kill 3 --jam --sim   # full stress test
```

### Scenarios

```bash
python3.10 main.py --scenario obstacle_run       # navigate through obstacles in V formation
python3.10 main.py --scenario formation_demo     # cycle through all formations
python3.10 main.py --scenario moving_obstacles   # dodge moving obstacles
python3.10 main.py --scenario dense_field        # 20 random obstacles, navigate to target
```

### Output

```bash
python3.10 main.py --log telemetry.json     # save telemetry snapshots
python3.10 main.py --report metrics.json    # save performance metrics
```

### Standalone Visualizer

`visualize.py` runs independently with its own built-in scenarios:

```bash
python3.10 visualize.py --drones 20
python3.10 visualize.py --scenario formation
python3.10 visualize.py --scenario dense
python3.10 visualize.py --speed 2
```

### All CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--drones` | 10 | Number of drones |
| `--sim` | off | Enable 3D visualization |
| `--duration` | 30.0 | Simulation duration (seconds) |
| `--tick-rate` | 30.0 | Simulation ticks per second |
| `--altitude` | 10.0 | Default hover altitude (meters) |
| `--max-speed` | 15.0 | Max drone speed (m/s) |
| `--formation` | none | Starting formation |
| `--scenario` | — | Preset scenario |
| `--mission` | — | Mission type |
| `--kill N` | 0 | Kill N drones at t=10s |
| `--jam` | off | Jam comms at t=15s |
| `--no-boids` | — | Disable flocking |
| `--no-consensus` | — | Disable consensus engine |
| `--no-comms` | — | Disable mesh network sim |
| `--no-metrics` | — | Disable metrics collection |
| `--log PATH` | — | Save telemetry JSON |
| `--report PATH` | — | Save metrics JSON |

---

## Programmatic API

```python
from swarm import SwarmManager, SwarmConfig, GeofencePlugin
from drone import PhysicsConfig
from formation import FormationType
from missions import AreaSearchMission
import numpy as np

# Configure
config = SwarmConfig(
    num_drones=20,
    physics=PhysicsConfig(max_speed=12.0),
    tick_rate=30.0,
)
swarm = SwarmManager(config)
swarm.plugins.register(GeofencePlugin())

# Add obstacles
swarm.add_obstacle(np.array([20.0, 0.0, 10.0]), radius=4.0)

# Takeoff and form up
swarm.takeoff_all()
swarm.set_formation(FormationType.V_SHAPE, center=np.array([0, 0, 10]))

# Assign a mission
mission = AreaSearchMission(np.array([0, 0, 10]), width=80, height=80)
swarm.mission_planner.assign_mission(mission, list(swarm.drones.keys()), swarm.drones)

# Run
swarm.run(duration=60)
print(swarm.status())
```

Custom tick loop:

```python
while running:
    swarm.tick()
    positions = swarm.get_positions()
    # render, log, stream, whatever
```

---

## Extending

### Custom Plugin

```python
from swarm import DronePlugin

class WindPlugin(DronePlugin):
    name = "wind"
    priority = 5

    def on_tick(self, drones, swarm_state):
        wind = np.array([2.0, 0.5, 0.0])
        for d in drones.values():
            d.add_plugin_vector("wind", wind * 0.1)

swarm.plugins.register(WindPlugin())
```

### Custom Formation

Subclass `FormationStrategy` in `formation.py` and return target positions for N drones.

### Custom Mission

Subclass `Mission` in `missions.py` and implement `plan()`, `tick()`, and `get_status()`.

### Custom Avoidance Layer

Subclass `ObstacleAvoidanceSystem` and override `reactive()`, `predictive()`, or `cooperative()`.

---

## Requirements

- Python 3.10+
- NumPy
- Matplotlib (only for `--sim` / `visualize.py`)

---

## Contributions

Utlized AI to write the comments, clean the codebase and write this README.
Everything else is my own work.

---

## License

MIT
