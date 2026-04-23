#!/usr/bin/env python3
"""
main.py — DroneSwarm AI Framework Entry Point
Runs the swarm with all systems: avoidance, formations, mesh comms,
decentralized consensus, missions, and performance metrics.

Basic usage:
    python3.10 main.py                                  # 10 drones, headless
    python3.10 main.py --drones 20 --sim                # 20 drones, 3D view
    python3.10 main.py --formation v_shape              # start in V formation
    python3.10 main.py --scenario obstacle_run          # preset scenario

Missions:
    python3.10 main.py --mission search                 # area search sweep
    python3.10 main.py --mission patrol                 # perimeter patrol
    python3.10 main.py --mission track                  # target tracking
    python3.10 main.py --mission relay                  # comm relay chain
    python3.10 main.py --mission all                    # split drones across all missions

Stress testing:
    python3.10 main.py --kill 4                         # kill 4 drones at t=10s
    python3.10 main.py --jam                            # jam comms at t=15s
    python3.10 main.py --mission all --kill 3 --jam     # full stress test

Output:
    python3.10 main.py --log telemetry.json             # save telemetry
    python3.10 main.py --report metrics.json            # save performance metrics
"""

import argparse
import numpy as np

from drone import PhysicsConfig
from avoidance import AvoidanceConfig
from formation import FormationType, BoidsConfig
from comms import NetworkConfig
from consensus import ConsensusConfig
from swarm import SwarmManager, SwarmConfig, GeofencePlugin, TelemetryPlugin, BatteryPlugin
from missions import (AreaSearchMission, PerimeterPatrolMission,
                      TargetTrackMission, RelayChainMission)


# ═══════════════════════════════════════════════════════════════════
# Scenarios (original presets)
# ═══════════════════════════════════════════════════════════════════

def scenario_obstacle_run(swarm):
    swarm.add_obstacle(np.array([20.0, 0.0, 10.0]), radius=4.0)
    swarm.add_obstacle(np.array([-15.0, 15.0, 8.0]), radius=3.0)
    swarm.add_obstacle(np.array([0.0, -20.0, 12.0]), radius=5.0)
    swarm.add_obstacle(np.array([30.0, 20.0, 10.0]), radius=3.5)
    swarm.add_obstacle(np.array([-25.0, -10.0, 15.0]), radius=2.5)
    swarm.takeoff_all()
    target = np.array([40.0, 30.0, 10.0])
    def on_tick(s):
        if abs(s.elapsed_time - 3.0) < s.dt:
            s.set_formation(FormationType.V_SHAPE, center=target)
            print("[Scenario] V-formation → navigating to target")
    swarm.on_tick(on_tick)


def scenario_formation_demo(swarm):
    swarm.takeoff_all()
    schedule = [
        (3.0, FormationType.CIRCLE, "Circle"),
        (8.0, FormationType.V_SHAPE, "V-Shape"),
        (13.0, FormationType.LINE, "Line"),
        (18.0, FormationType.GRID, "Grid"),
        (23.0, FormationType.SPIRAL, "Spiral"),
        (28.0, FormationType.DIAMOND, "Diamond"),
    ]
    def on_tick(s):
        for t, fmt, name in schedule:
            if abs(s.elapsed_time - t) < s.dt:
                print(f"[Scenario] → {name}")
                s.set_formation(fmt, center=np.array([0.0, 0.0, 10.0]))
    swarm.on_tick(on_tick)


def scenario_moving_obstacles(swarm):
    swarm.add_obstacle(np.array([-50.0, 0.0, 10.0]), 3.0, np.array([3.0, 0.0, 0.0]))
    swarm.add_obstacle(np.array([0.0, -50.0, 12.0]), 4.0, np.array([0.0, 2.5, 0.0]))
    swarm.add_obstacle(np.array([40.0, 40.0, 8.0]), 2.5, np.array([-2.0, -2.0, 0.0]))
    swarm.takeoff_all()
    def on_tick(s):
        if abs(s.elapsed_time - 2.0) < s.dt:
            s.set_formation(FormationType.CIRCLE, center=np.array([0.0, 0.0, 10.0]), radius=12.0)
    swarm.on_tick(on_tick)


def scenario_dense_field(swarm):
    rng = np.random.RandomState(42)
    for _ in range(20):
        pos = rng.uniform(-40, 40, 3).astype(float)
        pos[2] = rng.uniform(5, 20)
        swarm.add_obstacle(pos, rng.uniform(1.5, 4.0))
    swarm.takeoff_all()
    def on_tick(s):
        if abs(s.elapsed_time - 3.0) < s.dt:
            s.send_all_to(np.array([30.0, 30.0, 12.0]))
    swarm.on_tick(on_tick)


SCENARIOS = {
    "obstacle_run": scenario_obstacle_run,
    "formation_demo": scenario_formation_demo,
    "moving_obstacles": scenario_moving_obstacles,
    "dense_field": scenario_dense_field,
}


# ═══════════════════════════════════════════════════════════════════
# Mission Setup
# ═══════════════════════════════════════════════════════════════════

def setup_missions(swarm, mission_type):
    ids = list(swarm.drones.keys())
    n = len(ids)

    swarm.add_obstacle(np.array([20.0, 5.0, 10.0]), 4.0)
    swarm.add_obstacle(np.array([-15.0, -20.0, 10.0]), 5.0)
    swarm.add_obstacle(np.array([10.0, 25.0, 10.0]), 3.5)
    swarm.takeoff_all()

    if mission_type == "all":
        third = max(n // 3, 2)
        m1 = AreaSearchMission(np.array([0.0, 0.0, 10.0]), width=70, height=70, sweep_spacing=8.0)
        swarm.mission_planner.assign_mission(m1, ids[:third], swarm.drones)
        print(f"  area_search → {third} drones")
        m2 = PerimeterPatrolMission(np.array([0.0, 0.0, 10.0]), radius=30.0, orbit_speed=0.2)
        swarm.mission_planner.assign_mission(m2, ids[third:third*2], swarm.drones)
        print(f"  perimeter_patrol → {len(ids[third:third*2])} drones")
        m3 = TargetTrackMission(np.array([30.0, 30.0, 10.0]), np.array([-1.5, -0.8, 0.0]), standoff_radius=12.0)
        swarm.mission_planner.assign_mission(m3, ids[third*2:], swarm.drones)
        print(f"  target_track → {len(ids[third*2:])} drones")
    elif mission_type == "search":
        m = AreaSearchMission(np.array([0.0, 0.0, 10.0]), width=80, height=80, sweep_spacing=8.0)
        swarm.mission_planner.assign_mission(m, ids, swarm.drones)
        print(f"  area_search → {n} drones")
    elif mission_type == "patrol":
        m = PerimeterPatrolMission(np.array([0.0, 0.0, 10.0]), radius=30.0, orbit_speed=0.2)
        swarm.mission_planner.assign_mission(m, ids, swarm.drones)
        print(f"  perimeter_patrol → {n} drones")
    elif mission_type == "track":
        m = TargetTrackMission(np.array([30.0, 30.0, 10.0]), np.array([-1.5, -0.8, 0.0]), standoff_radius=12.0)
        swarm.mission_planner.assign_mission(m, ids, swarm.drones)
        print(f"  target_track → {n} drones")
    elif mission_type == "relay":
        m = RelayChainMission(np.array([-40.0, -30.0, 15.0]), np.array([40.0, 30.0, 15.0]))
        swarm.mission_planner.assign_mission(m, ids, swarm.drones)
        print(f"  relay_chain → {n} drones")


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def run_visual(swarm, duration):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    trails = {did: [] for did in swarm.drones}
    trail_len = 50
    colors = plt.cm.tab10(np.linspace(0, 1, len(swarm.drones)))
    total_frames = int(duration * swarm.config.tick_rate)
    ax.view_init(elev=28, azim=-55)

    def update(frame):
        swarm.tick()
        elev, azim = ax.elev, ax.azim
        ax.clear()
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60); ax.set_zlim(0, 50)
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

        for g in np.linspace(-60, 60, 13):
            ax.plot([g, g], [-60, 60], [0, 0], color='#CCC', linewidth=0.3, alpha=0.5)
            ax.plot([-60, 60], [g, g], [0, 0], color='#CCC', linewidth=0.3, alpha=0.5)

        for obs in swarm.obstacles:
            u, v = np.linspace(0, 2*np.pi, 10), np.linspace(0, np.pi, 6)
            p, r = obs.position, obs.radius
            ax.plot_wireframe(
                p[0]+r*np.outer(np.cos(u), np.sin(v)),
                p[1]+r*np.outer(np.sin(u), np.sin(v)),
                p[2]+r*np.outer(np.ones_like(u), np.cos(v)),
                color='#CC3333', alpha=0.35, linewidth=0.5)

        for i, (did, d) in enumerate(swarm.drones.items()):
            if did not in trails: trails[did] = []
            trails[did].append(d.position.copy())
            if len(trails[did]) > trail_len: trails[did] = trails[did][-trail_len:]
            ax.scatter(*d.position, color=colors[i % len(colors)], s=60, marker='^',
                       edgecolors='#444', linewidths=0.3, depthshade=True)
            if np.linalg.norm(d.velocity) > 0.3:
                ax.quiver(*d.position, *(d.velocity*0.4), color='#555', alpha=0.4,
                          arrow_length_ratio=0.3, linewidth=0.6)
            t = np.array(trails[did])
            if len(t) > 1:
                ax.plot(t[:,0], t[:,1], t[:,2], color='#888', alpha=0.15, linewidth=0.8)

        modes = {}
        for d in swarm.drones.values():
            modes[d.mode.value] = modes.get(d.mode.value, 0) + 1
        leader = swarm.consensus.leader_id if swarm.consensus else "-"
        health = swarm.consensus.swarm_health.value if swarm.consensus else "-"
        ax.set_title(
            f"{len(swarm.drones)} drones | t={swarm.elapsed_time:.1f}s | "
            f"{' '.join(f'{m}:{c}' for m,c in modes.items())}\n"
            f"Leader:{leader}  Health:{health}  |  Drag to rotate",
            fontsize=8, fontfamily="monospace", color="#444", pad=10)

    fig.canvas.mpl_connect("key_press_event",
        lambda e: plt.close(fig) if e.key == 'q' else None)
    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=1000/swarm.config.tick_rate, blit=False)
    plt.show()


def run_headless(swarm, duration):
    ticks = int(duration * swarm.config.tick_rate)
    report = int(swarm.config.tick_rate)  # every 1 second

    for i in range(ticks):
        swarm.tick()
        if (i + 1) % report == 0:
            t = swarm.elapsed_time
            n = len(swarm.drones)
            leader = swarm.consensus.leader_id if swarm.consensus else "-"
            health = swarm.consensus.swarm_health.value if swarm.consensus else "-"
            delivery = f"{swarm.network.delivery_rate:.0%}" if swarm.network else "-"
            missions = len(swarm.mission_planner.get_active_missions())
            modes = {}
            for d in swarm.drones.values():
                modes[d.mode.value] = modes.get(d.mode.value, 0) + 1
            mode_str = " ".join(f"{m}:{c}" for m, c in modes.items())
            print(f"  [{t:5.1f}s] {n} drones | {mode_str} | "
                  f"leader:{leader} health:{health} comms:{delivery} missions:{missions}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DroneSwarm AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:  obstacle_run, formation_demo, moving_obstacles, dense_field
Missions:   search, patrol, track, relay, all
        """)
    parser.add_argument("--drones", type=int, default=10)
    parser.add_argument("--sim", action="store_true", help="3D visualization")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--tick-rate", type=float, default=30.0)
    parser.add_argument("--altitude", type=float, default=10.0)
    parser.add_argument("--max-speed", type=float, default=15.0)
    parser.add_argument("--formation", default="none", choices=[f.value for f in FormationType])
    parser.add_argument("--scenario", default=None, choices=list(SCENARIOS.keys()))
    parser.add_argument("--mission", default=None, choices=["search", "patrol", "track", "relay", "all"])
    parser.add_argument("--no-boids", action="store_true")
    parser.add_argument("--no-consensus", action="store_true")
    parser.add_argument("--no-comms", action="store_true")
    parser.add_argument("--no-metrics", action="store_true")
    parser.add_argument("--kill", type=int, default=0, help="Kill N drones at t=10s")
    parser.add_argument("--jam", action="store_true", help="Jam comms at t=15s")
    parser.add_argument("--log", default=None, help="Save telemetry JSON")
    parser.add_argument("--report", default=None, help="Save metrics JSON")
    args = parser.parse_args()

    config = SwarmConfig(
        num_drones=args.drones,
        physics=PhysicsConfig(max_speed=args.max_speed),
        tick_rate=args.tick_rate,
        default_altitude=args.altitude,
        formation=FormationType(args.formation),
        enable_boids=not args.no_boids,
        enable_consensus=not args.no_consensus,
        enable_comms_sim=not args.no_comms,
        enable_metrics=not args.no_metrics,
    )
    swarm = SwarmManager(config)
    swarm.plugins.register(GeofencePlugin())
    telemetry = TelemetryPlugin(interval=10)
    swarm.plugins.register(telemetry)

    if args.scenario:
        SCENARIOS[args.scenario](swarm)
    elif args.mission:
        setup_missions(swarm, args.mission)
    else:
        swarm.takeoff_all()
        if config.formation != FormationType.NONE:
            def auto_form(s):
                if abs(s.elapsed_time - 3.0) < s.dt:
                    s.set_formation(config.formation, center=np.array([0.0, 0.0, config.default_altitude]))
            swarm.on_tick(auto_form)

    # Stress hooks
    kill_done = False
    jam_done = False
    def stress_hook(s):
        nonlocal kill_done, jam_done
        if args.kill > 0 and s.elapsed_time >= 10.0 and not kill_done:
            kill_done = True
            for did in list(s.drones.keys())[:args.kill]:
                s.remove_drone(did)
            print(f"\n  ⚠ KILLED {args.kill} drones at t={s.elapsed_time:.1f}s → {len(s.drones)} remaining")
        if args.jam and s.elapsed_time >= 15.0 and not jam_done:
            jam_done = True
            if s.network:
                s.network.config.interference_factor = 0.6
            print(f"\n  ⚠ COMMS JAMMING at t={s.elapsed_time:.1f}s")
    if args.kill > 0 or args.jam:
        swarm.on_tick(stress_hook)

    mode_label = args.scenario or args.mission or args.formation
    print(f"{'='*60}")
    print(f"  DroneSwarm AI | {config.num_drones} drones | {config.tick_rate} Hz | {args.duration}s")
    print(f"  Mode: {mode_label} | Obstacles: {len(swarm.obstacles)}")
    if args.kill: print(f"  Stress: kill {args.kill} at t=10s")
    if args.jam: print(f"  Stress: jam comms at t=15s")
    print(f"{'='*60}")

    if args.sim:
        run_visual(swarm, args.duration)
    else:
        run_headless(swarm, args.duration)

    # Final
    print(f"\n{'='*60}")
    print(f"  FINAL STATE")
    print(f"{'='*60}")
    print(f"  {swarm.status()}")
    if swarm.mission_planner.missions:
        print(f"\n  MISSIONS:")
        for m in swarm.mission_planner.missions:
            print(f"    {m.get_status()}")
    if args.log:
        telemetry.save(args.log)
        print(f"\n  Telemetry → {args.log}")
    if swarm.metrics:
        swarm.metrics.print_summary()
        if args.report:
            swarm.metrics.export(args.report)
            print(f"  Metrics → {args.report}")


if __name__ == "__main__":
    main()
