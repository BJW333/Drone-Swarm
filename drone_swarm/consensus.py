"""
consensus.py — Decentralized Consensus & Autonomous Operations
No central controller. Drones elect leaders, vote on swarm state,
and operate independently when communications degrade.

Key features:
    - Bully-algorithm leader election (highest-capability drone wins)
    - Heartbeat-based failure detection
    - Swarm health consensus (drones vote on swarm state)
    - Autonomous fallback behaviors when isolated
    - Role assignment (leader, wingman, scout, relay)
    - Graceful degradation — swarm keeps operating as drones drop out
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from drone import DroneAgent, DroneMode, dist
from comms import MeshNetwork, NetMessage, MessagePriority


# ═══════════════════════════════════════════════════════════════════
# Roles & State
# ═══════════════════════════════════════════════════════════════════

class DroneRole(Enum):
    LEADER = "leader"
    WINGMAN = "wingman"
    SCOUT = "scout"
    RELAY = "relay"
    AUTONOMOUS = "autonomous"  # lost contact, operating alone


class SwarmHealthState(Enum):
    NOMINAL = "nominal"           # all systems go
    DEGRADED = "degraded"         # some drones lost, still functional
    CRITICAL = "critical"         # below minimum viable swarm
    PARTITIONED = "partitioned"   # network split


@dataclass
class ConsensusConfig:
    election_timeout: float = 3.0       # seconds to wait for election response
    health_vote_interval: float = 2.0   # how often drones vote on swarm health
    isolation_timeout: float = 5.0      # seconds before drone considers itself isolated
    min_viable_swarm: int = 3           # below this = CRITICAL
    leader_heartbeat: float = 1.0       # leader broadcasts state this often
    role_reassign_interval: float = 5.0 # re-evaluate roles this often


@dataclass
class DroneConsensusState:
    """Per-drone consensus data."""
    role: DroneRole = DroneRole.WINGMAN
    leader_id: Optional[str] = None
    last_leader_heartbeat: float = 0.0
    known_alive: Dict[str, float] = field(default_factory=dict)  # drone_id -> last_heard
    health_vote: SwarmHealthState = SwarmHealthState.NOMINAL
    capability_score: float = 0.0  # battery * connectivity, used for elections
    is_isolated: bool = False
    election_in_progress: bool = False
    election_started_at: float = 0.0
    votes_received: Dict[str, str] = field(default_factory=dict)  # voter -> candidate


# ═══════════════════════════════════════════════════════════════════
# Consensus Engine
# ═══════════════════════════════════════════════════════════════════

class ConsensusEngine:
    """
    Runs decentralized consensus across the swarm.
    Each drone maintains its own view of the swarm and makes local decisions.

    Attach to SwarmManager:
        consensus = ConsensusEngine(config)
        consensus.init(drones, network)
        # each tick:
        consensus.tick(drones, network, dt)
    """

    def __init__(self, config: ConsensusConfig = None):
        self.config = config or ConsensusConfig()
        self.states: Dict[str, DroneConsensusState] = {}
        self._sim_time: float = 0.0
        self._last_health_vote: float = 0.0
        self._last_role_assign: float = 0.0
        self.swarm_health: SwarmHealthState = SwarmHealthState.NOMINAL
        self.leader_id: Optional[str] = None

    def init(self, drones: Dict[str, DroneAgent], network: MeshNetwork):
        """Initialize consensus state for all drones."""
        for did, drone in drones.items():
            score = drone.battery_pct / 100.0
            self.states[did] = DroneConsensusState(
                capability_score=score,
                known_alive={d: 0.0 for d in drones if d != did},
            )
        # Initial leader = highest capability
        if drones:
            self.leader_id = max(self.states, key=lambda d: self.states[d].capability_score)
            self.states[self.leader_id].role = DroneRole.LEADER
            for did in self.states:
                self.states[did].leader_id = self.leader_id

    def tick(self, drones: Dict[str, DroneAgent], network: MeshNetwork, dt: float):
        """Run one consensus tick. Call every sim tick."""
        self._sim_time += dt

        # Remove states for drones that no longer exist
        dead = [did for did in self.states if did not in drones]
        for did in dead:
            del self.states[did]
            # If leader died, immediately elect new leader from survivors
            if did == self.leader_id:
                self.leader_id = None
                if self.states:
                    new_leader = max(self.states, key=lambda d: self.states[d].capability_score)
                    self._declare_leader(new_leader, drones, network)

        for did, drone in drones.items():
            if did not in self.states:
                continue
            state = self.states[did]

            # Update capability score
            neighbors = network.get_neighbors(did)
            connectivity = len(neighbors) / max(len(drones) - 1, 1)
            state.capability_score = (drone.battery_pct / 100.0) * 0.6 + connectivity * 0.4

            # Update known_alive from network
            for nid in neighbors:
                state.known_alive[nid] = self._sim_time
            alive_count = sum(
                1 for t in state.known_alive.values()
                if self._sim_time - t < self.config.isolation_timeout
            )

            # Isolation detection
            state.is_isolated = len(neighbors) == 0
            if state.is_isolated:
                state.role = DroneRole.AUTONOMOUS
                self._apply_autonomous_behavior(drone)
                continue

            # Leader failure detection
            if state.leader_id and state.leader_id != did:
                leader_alive = state.leader_id in neighbors or (
                    state.leader_id in state.known_alive and
                    self._sim_time - state.known_alive.get(state.leader_id, 0) < self.config.election_timeout
                )
                if not leader_alive and not state.election_in_progress:
                    self._start_election(did, drones, network)
            elif self.leader_id is None and not state.election_in_progress:
                # No leader at all — start election
                self._start_election(did, drones, network)

            # Leader duties
            if state.role == DroneRole.LEADER:
                self._leader_tick(did, drone, drones, network, state)

        # Periodic health vote
        if self._sim_time - self._last_health_vote >= self.config.health_vote_interval:
            self._health_consensus(drones, network)
            self._last_health_vote = self._sim_time

        # Periodic role reassignment
        if self._sim_time - self._last_role_assign >= self.config.role_reassign_interval:
            self._assign_roles(drones, network)
            self._last_role_assign = self._sim_time

    # ── Leader Election (Bully Algorithm) ──

    def _start_election(self, initiator_id: str, drones: Dict, network: MeshNetwork):
        """Initiate leader election. Highest capability score wins."""
        state = self.states[initiator_id]
        state.election_in_progress = True
        state.election_started_at = self._sim_time
        state.votes_received = {initiator_id: initiator_id}

        # Broadcast election message
        network.send(NetMessage(
            sender_id=initiator_id,
            msg_type="election_start",
            payload={
                "candidate": initiator_id,
                "score": state.capability_score,
            },
            priority=MessagePriority.COMMAND,
        ))

        # Check if anyone with higher score is reachable
        neighbors = network.get_neighbors(initiator_id)
        higher_exists = False
        for nid in neighbors:
            if nid in self.states and self.states[nid].capability_score > state.capability_score:
                higher_exists = True
                break

        if not higher_exists:
            # I win — declare leadership
            self._declare_leader(initiator_id, drones, network)

    def _declare_leader(self, leader_id: str, drones: Dict, network: MeshNetwork):
        """Announce new leader to swarm."""
        self.leader_id = leader_id
        for did in self.states:
            self.states[did].leader_id = leader_id
            self.states[did].election_in_progress = False
            if did == leader_id:
                self.states[did].role = DroneRole.LEADER
            elif self.states[did].role == DroneRole.LEADER:
                self.states[did].role = DroneRole.WINGMAN

        network.send(NetMessage(
            sender_id=leader_id,
            msg_type="leader_announce",
            payload={"leader": leader_id},
            priority=MessagePriority.COMMAND,
        ))

    # ── Leader Duties ──

    def _leader_tick(self, did: str, drone: DroneAgent, drones: Dict,
                     network: MeshNetwork, state: DroneConsensusState):
        """What the leader does each tick."""
        # Broadcast state to swarm
        if self._sim_time - state.last_leader_heartbeat >= self.config.leader_heartbeat:
            state.last_leader_heartbeat = self._sim_time

            alive_drones = [d for d in self.states
                            if not self.states[d].is_isolated and d in drones]

            network.send(NetMessage(
                sender_id=did,
                msg_type="leader_state",
                payload={
                    "leader": did,
                    "swarm_health": self.swarm_health.value,
                    "alive_count": len(alive_drones),
                    "time": self._sim_time,
                },
                priority=MessagePriority.COMMAND,
            ))

    # ── Health Consensus ──

    def _health_consensus(self, drones: Dict, network: MeshNetwork):
        """Each drone votes on swarm health, majority wins."""
        votes = {}
        for did, state in self.states.items():
            if state.is_isolated:
                continue

            neighbors = network.get_neighbors(did)
            alive = len(neighbors) + 1  # +1 for self
            total = len(drones)

            if alive >= total * 0.8:
                vote = SwarmHealthState.NOMINAL
            elif alive >= total * 0.5:
                vote = SwarmHealthState.DEGRADED
            elif alive >= self.config.min_viable_swarm:
                vote = SwarmHealthState.CRITICAL
            else:
                vote = SwarmHealthState.CRITICAL

            state.health_vote = vote
            votes[vote] = votes.get(vote, 0) + 1

        # Check for network partition
        if network.is_partitioned():
            self.swarm_health = SwarmHealthState.PARTITIONED
        elif votes:
            self.swarm_health = max(votes, key=votes.get)

    # ── Role Assignment ──

    def _assign_roles(self, drones: Dict, network: MeshNetwork):
        """Leader assigns roles based on position and connectivity."""
        if not self.leader_id or self.leader_id not in self.states:
            return

        active = [did for did in self.states
                   if not self.states[did].is_isolated and did in drones]

        if len(active) < 2:
            return

        # Sort by capability
        ranked = sorted(active, key=lambda d: self.states[d].capability_score, reverse=True)

        for i, did in enumerate(ranked):
            state = self.states[did]
            if did == self.leader_id:
                state.role = DroneRole.LEADER
            elif i < len(ranked) * 0.2:
                # Top 20% by capability = scouts (farthest out)
                state.role = DroneRole.SCOUT
            elif network.get_neighbors(did) and len(network.get_neighbors(did)) >= len(active) * 0.5:
                # Well-connected drones = relays
                state.role = DroneRole.RELAY
            else:
                state.role = DroneRole.WINGMAN

    # ── Autonomous Fallback ──

    def _apply_autonomous_behavior(self, drone: DroneAgent):
        """When a drone loses all comms, it falls back to safe behavior."""
        if drone.mode == DroneMode.EMERGENCY:
            return

        # If battery critical, land
        if drone.battery_pct < 15:
            drone.land()
            return

        # Otherwise, hold position and slowly orbit to try to regain comms
        if drone.mode not in (DroneMode.LANDING, DroneMode.IDLE):
            # Orbit at current position to scan for neighbors
            t = self._sim_time
            orbit_radius = 10.0
            orbit_target = drone.position.copy()
            orbit_target[0] += orbit_radius * np.cos(t * 0.3)
            orbit_target[1] += orbit_radius * np.sin(t * 0.3)
            drone.target = orbit_target
            drone.mode = DroneMode.NAVIGATE

    # ── Message Handler ──

    def handle_message(self, drone_id: str, msg: NetMessage):
        """Process consensus-related messages. Wire this up to network callbacks."""
        if drone_id not in self.states:
            return

        state = self.states[drone_id]

        if msg.msg_type == "election_start":
            candidate = msg.payload.get("candidate")
            score = msg.payload.get("score", 0)
            # If I have higher score, I should be leader
            if state.capability_score > score:
                state.election_in_progress = True
                # In a full implementation, would send "election_response" back
            else:
                state.leader_id = candidate

        elif msg.msg_type == "leader_announce":
            new_leader = msg.payload.get("leader")
            state.leader_id = new_leader
            state.election_in_progress = False
            if drone_id != new_leader and state.role == DroneRole.LEADER:
                state.role = DroneRole.WINGMAN

        elif msg.msg_type == "leader_state":
            state.last_leader_heartbeat = self._sim_time
            state.leader_id = msg.payload.get("leader")

        # Track who's alive
        state.known_alive[msg.sender_id] = self._sim_time

    # ── Queries ──

    def get_role(self, drone_id: str) -> DroneRole:
        if drone_id in self.states:
            return self.states[drone_id].role
        return DroneRole.WINGMAN

    def get_leader(self) -> Optional[str]:
        return self.leader_id

    def get_alive_count(self) -> int:
        return sum(1 for s in self.states.values() if not s.is_isolated)

    def get_isolated(self) -> List[str]:
        return [did for did, s in self.states.items() if s.is_isolated]

    def status(self) -> str:
        roles = {}
        for s in self.states.values():
            roles[s.role.value] = roles.get(s.role.value, 0) + 1
        isolated = len(self.get_isolated())
        return (f"Health:{self.swarm_health.value} Leader:{self.leader_id} "
                f"Alive:{self.get_alive_count()} Isolated:{isolated} Roles:{roles}")
