#!/usr/bin/env python3
"""
================================================================================
SOCIAL FLOCKING VIA EMOTION-LIKE CONTROL
================================================================================

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

================================================================================

Demonstrates spontaneous flock formation from random initial conditions using
the affect-based control architecture (A1-A8), WITHOUT episodic memory.

Key Insight:
    Heading coherence is treated as a NEED, not just a mechanical rule.
    When an agent's heading differs from neighbors, it experiences discomfort
    ("out of sync"), which drives the Align policy. This is biologically
    plausible—social animals experience discomfort when not coordinating
    with the group.

Needs System:
    - Affiliation: Desire to be near others (prevents isolation)
    - Safety: Desire for personal space (prevents collision)
    - Motion: Desire to keep moving (flying birds can't hover)
    - Coherence: Desire to align with neighbors (social coordination)

Policies:
    - Seek: Move toward neighbors (activated when lonely)
    - Avoid: Move away from too-close neighbors (activated when crowded)
    - Align: Match neighbor headings (activated when misaligned)
    - Explore: Random movement (activated when uncertain)

Algorithm Mapping to Paper (A1-A8):
    A1 (Categorize): categorize() → [density, crowding, coherence, motion]
    A2 (Need Appraisal): NeedSystem.assess() and compute_affect()
        - Drives: d_i = tanh(α_i × (n_i - n_target_i))
        - Affect: z_t = [valence, magnitude, arousal, drives]
    A3 (Episodic Retrieval): Not implemented in this demo (stateless)
    A4 (Policy Mix): PolicySystem.get_policy_mix()
        - h_t(π) = α_need × h_need(π) + α_aff × h_aff(π)
        - q_t(π) = softmax(h_t / τ(arousal))
    A5 (Policy Instantiation): PolicySystem.score_actions()
        - s_t(u) = Σ_π q_t(π) × s̃_π(u)
    A6 (Action Selection): softmax over action scores
    A7 (Execute): Update position based on selected heading

Usage:
    python flock_no_memory.py

Outputs:
    - flock_trajectories.png: Agent paths colored by time
    - flock_snapshots.png: Flock configuration at key timesteps
    - flock_metrics.png: Alignment, cohesion, arousal, policy mix
    - flock_animation.mp4: Animation of flock formation

Reference:
    "Synthetic Emotions and Consciousness: Exploring Architectural Boundaries"

Repository:
    https://github.com/affect-based-control/synthetic-emotion-controller
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from matplotlib.cm import viridis, coolwarm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# UTILITIES
# ============================================================================

def wrap_angle(a: float) -> float:
    """Wrap angle to (-π, π]"""
    return math.atan2(math.sin(a), math.cos(a))

def ang_diff(a: float, b: float) -> float:
    """Signed angular difference (a - b), wrapped to (-π, π]"""
    return wrap_angle(a - b)

def clip(x: float, lo: float, hi: float) -> float:
    """Clip value to range [lo, hi]"""
    return max(lo, min(hi, x))

def softmax(x: np.ndarray, tau: float) -> np.ndarray:
    """
    Numerically stable softmax with temperature.
    Lower tau = more decisive (winner-take-all)
    Higher tau = more uniform (exploratory)
    """
    tau = max(1e-6, tau)
    x_shifted = x - np.max(x)  # numerical stability
    e = np.exp(x_shifted / tau)
    return e / (np.sum(e) + 1e-12)

def circular_mean(angles: List[float], weights: Optional[List[float]] = None) -> Tuple[float, float]:
    """
    Compute weighted circular mean of angles.
    
    Returns:
        (mean_angle, concentration) where concentration ∈ [0,1]
        concentration = 0 means random/uniform, 1 means perfect alignment
    """
    if len(angles) == 0:
        return 0.0, 0.0
    if weights is None:
        weights = [1.0] * len(angles)
    
    total_w = sum(weights) + 1e-12
    sx = sum(w * math.cos(a) for w, a in zip(weights, angles)) / total_w
    sy = sum(w * math.sin(a) for w, a in zip(weights, angles)) / total_w
    
    concentration = math.hypot(sx, sy)  # resultant length
    mean_angle = math.atan2(sy, sx)
    
    return mean_angle, clip(concentration, 0, 1)

# ============================================================================
# AGENT STATE
# ============================================================================

@dataclass
class AgentState:
    """State of a single flocking agent"""
    x: float          # position x
    y: float          # position y  
    theta: float      # heading (radians)
    speed: float      # current speed
    
    # For visualization
    color_id: int = 0  # agent identifier for coloring

# ============================================================================
# A1: CATEGORIZATION
# ============================================================================

def categorize(agent: AgentState, peers: List[Tuple[float, float, float]], 
               R_comfort: float = 5.0, R_too_close: float = 1.5) -> Dict:
    """
    A1: Classify the current situation into category vector c_t.
    
    This is the "perceptual preprocessor" that maps raw observations
    to a compact situational description.
    
    Args:
        agent: Current agent state
        peers: List of (distance, bearing, heading) for each neighbor
        R_comfort: Comfortable interaction distance
        R_too_close: Personal space violation distance
        
    Returns:
        Dictionary with:
        - 'c': Category vector [density, crowding, coherence, motion]
        - 'y': Parameters for policy instantiation
    """
    # Handle empty neighborhood
    if len(peers) == 0:
        return {
            'c': np.array([0.0, 0.0, 0.5, agent.speed]),
            'y': {
                'mean_neighbor_dir': None,
                'mean_neighbor_heading': None,
                'nearest_dir': None,
                'nearest_dist': float('inf'),
            }
        }
    
    distances = [r for (r, phi, h) in peers]
    bearings = [phi for (r, phi, h) in peers]
    headings = [h for (r, phi, h) in peers]
    
    # ---- Category: Density (c_density) ----
    # How many neighbors are within comfortable distance?
    # Maps to affiliation need fulfillment
    near_count = sum(1 for r in distances if r < R_comfort)
    c_density = clip(near_count / 5.0, 0, 1)  # 5+ neighbors = fully "affiliated"
    
    # ---- Category: Crowding (c_too_close) ----
    # Is anyone violating personal space?
    # Maps to safety need threat
    min_dist = min(distances)
    if min_dist < R_too_close:
        c_too_close = clip(1.0 - min_dist / R_too_close, 0, 1)
    else:
        c_too_close = 0.0
    
    # ---- Category: Heading Coherence (c_coherence) ----
    # Are nearby neighbors heading the same direction as me?
    # This is KEY for flocking - creates emotional response to misalignment
    nearby_mask = [r < R_comfort * 1.5 for (r, phi, h) in peers]
    nearby_headings = [h for h, m in zip(headings, nearby_mask) if m]
    
    if len(nearby_headings) > 0:
        # Include my own heading in coherence calculation
        all_headings = [agent.theta] + nearby_headings
        _, coherence = circular_mean(all_headings)
        c_coherence = coherence
    else:
        c_coherence = 0.5  # neutral when alone
    
    # ---- Category: Motion (c_moving) ----
    # Am I currently moving?
    c_moving = clip(agent.speed, 0, 1)
    
    # ---- Parameters for Policy Instantiation (y_t) ----
    # These are situation-specific values used by policy templates
    
    # Direction to neighbors (weighted by proximity) - for Seek
    proximity_weights = [math.exp(-r / R_comfort) for r in distances]
    mean_neighbor_dir, _ = circular_mean(bearings, proximity_weights)
    
    # Mean heading of nearby neighbors - for Align
    if len(nearby_headings) > 0:
        mean_neighbor_heading, _ = circular_mean(nearby_headings)
    else:
        mean_neighbor_heading = agent.theta
    
    # Nearest neighbor direction - for Avoid
    nearest_idx = distances.index(min_dist)
    nearest_dir = bearings[nearest_idx]
    
    return {
        'c': np.array([c_density, c_too_close, c_coherence, c_moving]),
        'y': {
            'mean_neighbor_dir': mean_neighbor_dir,
            'mean_neighbor_heading': mean_neighbor_heading,
            'nearest_dir': nearest_dir,
            'nearest_dist': min_dist,
        }
    }

# ============================================================================
# A2: NEED SYSTEM AND AFFECT
# ============================================================================

class NeedSystem:
    """
    A2: Assess needs and compute affect.
    
    Four needs for flocking:
    1. Affiliation - desire to be near others (prevents isolation)
    2. Safety - desire for personal space (prevents collision)
    3. Motion - desire to keep moving (flying birds can't hover)
    4. Coherence - desire to align with neighbors (social coordination)
    
    Each need has a target level. Deviation from target creates DRIVES,
    which generate AFFECT (valence, magnitude, arousal).
    """
    
    def __init__(self, cfg: Dict):
        # Need targets (desired fulfillment levels)
        self.targets = np.array([
            cfg.get('n_affiliation_target', 0.5),   # want moderate density
            cfg.get('n_safety_target', 0.1),        # want low crowding (high safety)
            cfg.get('n_motion_target', 0.8),        # want to be moving
            cfg.get('n_coherence_target', 0.7),     # want alignment with group
        ])
        
        # Drive sensitivity (how strongly deviation affects drives)
        # Higher = more emotional response to need deviation
        self.alpha = np.array(cfg.get('drive_alpha', [2.5, 3.0, 2.0, 3.0]))
        
        # Base arousal (background activation level)
        self.base_arousal = cfg.get('base_arousal', 0.4)
        
    def assess(self, c: np.ndarray) -> np.ndarray:
        """
        Map category vector to need fulfillment levels.
        
        c = [c_density, c_too_close, c_coherence, c_moving]
        n = [n_affiliation, n_safety, n_motion, n_coherence]
        """
        n_affiliation = c[0]      # density → affiliation fulfilled
        n_safety = 1.0 - c[1]     # NOT crowded → safety fulfilled
        n_motion = c[3]           # moving → motion need fulfilled
        n_coherence = c[2]        # aligned → coherence fulfilled
        
        return np.array([n_affiliation, n_safety, n_motion, n_coherence])
    
    def compute_affect(self, n: np.ndarray) -> Dict:
        """
        Compute affect from need fulfillment.
        
        This implements:
            d_i = tanh(α_i × (n_i - n_target_i))
            
        Where d_i is the drive for need i:
        - Negative drive = need unfulfilled (below target)
        - Positive drive = need over-fulfilled (above target)
        
        Returns affect vector z_t = [valence, magnitude, arousal, drives]
        """
        # Drives: signed deviation from target
        raw_deviation = n - self.targets
        drives = np.tanh(self.alpha * raw_deviation)
        
        # Valence: emotional tone (positive/negative)
        # For these needs, valence matches drive sign
        # (unfulfilled = negative valence = feels bad)
        valence = drives.copy()
        
        # Magnitude: emotional intensity (always positive)
        magnitude = np.abs(drives)
        
        # Arousal: overall activation level
        # Increases when drives are strong (needs unmet)
        mean_magnitude = np.mean(magnitude)
        arousal = clip(self.base_arousal + 0.4 * mean_magnitude, 0, 1)
        
        return {
            'valence': valence,      # [4] signed
            'magnitude': magnitude,   # [4] unsigned  
            'arousal': arousal,       # scalar
            'drives': drives,         # [4] signed
            'needs': n,               # [4] fulfillment levels
            # Full affect vector for storage
            'z': np.concatenate([valence, magnitude, [arousal], drives])
        }

# ============================================================================
# A4-A5: POLICY SYSTEM
# ============================================================================

class PolicySystem:
    """
    A4: Compute policy mix from affect.
    A5: Instantiate policies into action scores.
    
    Policies for flocking:
    - Seek: Move toward neighbors (activated when lonely)
    - Avoid: Move away from too-close neighbors (activated when crowded)
    - Align: Match neighbor headings (activated when misaligned)
    - Explore: Random movement (activated when uncertain)
    
    Each policy has a TEMPLATE that scores actions based on current situation.
    The policy MIX q(π) determines how much each template contributes.
    """
    
    def __init__(self, cfg: Dict):
        self.policies = ['Seek', 'Avoid', 'Align', 'Explore']
        self.n_policies = len(self.policies)
        
        # Action space: discrete headings
        self.n_headings = cfg.get('n_headings', 36)
        self.headings = np.linspace(-math.pi, math.pi, self.n_headings, endpoint=False)
        
        # Von Mises concentration for action scoring
        self.kappa = cfg.get('kappa', 5.0)
        
        # ---- H_n: Need-to-Policy Mapping ----
        # Maps drives [affil, safety, motion, coherence] to policy hints
        # Negative weight means: low drive → high policy activation
        self.H_n = np.array(cfg.get('H_n', [
            # Seek: activated by LOW affiliation (lonely)
            [-1.2,  0.0,  0.2, -0.1],
            # Avoid: activated by LOW safety (too close to others)  
            [ 0.1, -1.5,  0.0,  0.0],
            # Align: activated by LOW coherence (heading mismatch)
            [-0.1,  0.0,  0.1, -1.3],
            # Explore: mild activation when motion is low
            [ 0.2,  0.2, -0.3,  0.2],
        ]))
        
        # ---- Fusion Weights ----
        self.alpha_need = cfg.get('alpha_need', 0.65)  # weight for need-based hints
        self.alpha_aff = cfg.get('alpha_aff', 0.35)    # weight for affect-based hints
        
        # ---- Temperature Schedule ----
        # Maps arousal to decision temperature
        # High arousal → low temperature → decisive (sharp distribution)
        # Low arousal → high temperature → exploratory (flat distribution)
        self.tau_high = cfg.get('tau_high', 0.6)  # temperature at low arousal
        self.tau_low = cfg.get('tau_low', 0.15)   # temperature at high arousal
        
    def compute_hints_from_needs(self, drives: np.ndarray) -> np.ndarray:
        """
        Compute policy hints from drives using H_n matrix.
        
        h_need(π) = [H_n × d]_π
        """
        hints = self.H_n @ drives
        # Normalize to [-1, 1] for consistent scale
        max_abs = np.abs(hints).max()
        if max_abs > 1e-6:
            hints = hints / max_abs
        return hints
    
    def compute_hints_from_affect(self, affect: Dict) -> np.ndarray:
        """
        Compute policy hints directly from affect state.
        
        This provides a secondary pathway from emotion to behavior,
        beyond the structured H_n mapping.
        """
        v = affect['valence']
        a = affect['arousal']
        
        hints = np.array([
            # Seek: negative affiliation valence + arousal boost
            -0.3 * v[0] + 0.2 * a,
            # Avoid: negative safety valence + strong arousal boost
            -0.4 * v[1] + 0.3 * a,
            # Align: negative coherence valence
            -0.5 * v[3] + 0.1 * a,
            # Explore: inverse arousal (explore when calm)
            -0.2 * a + 0.1 * np.mean(v),
        ])
        
        max_abs = np.abs(hints).max()
        if max_abs > 1e-6:
            hints = hints / max_abs
        return hints
    
    def get_policy_mix(self, affect: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        A4: Fuse hints and compute policy probabilities q(π).
        
        h(π) = α_need × h_need(π) + α_aff × h_aff(π)
        q(π) = softmax(h(π) / τ(arousal))
        
        Returns:
            q: Policy probabilities [n_policies]
            h: Fused hints [n_policies] (for logging)
        """
        drives = affect['drives']
        arousal = affect['arousal']
        
        # Compute hints from both sources
        h_need = self.compute_hints_from_needs(drives)
        h_aff = self.compute_hints_from_affect(affect)
        
        # Fuse hints (weighted combination)
        h = self.alpha_need * h_need + self.alpha_aff * h_aff
        
        # Temperature from arousal
        tau = self.tau_high - (self.tau_high - self.tau_low) * arousal
        
        # Convert to probabilities
        q = softmax(h, tau)
        
        return q, h
    
    def score_actions(self, q: np.ndarray, y: Dict, agent: AgentState) -> np.ndarray:
        """
        A5: Score each action using policy-weighted templates.
        
        s(u) = Σ_π q(π) × s̃_π(u)
        
        Each policy template s̃_π(u) scores how well action u
        implements policy π in the current situation.
        """
        scores = np.zeros(self.n_headings)
        
        for i, theta in enumerate(self.headings):
            # Get template scores for this heading
            s_seek = self._template_seek(theta, y, agent)
            s_avoid = self._template_avoid(theta, y, agent)
            s_align = self._template_align(theta, y, agent)
            s_explore = self._template_explore(theta, agent)
            
            # Policy-weighted combination
            scores[i] = (q[0] * s_seek + 
                        q[1] * s_avoid + 
                        q[2] * s_align + 
                        q[3] * s_explore)
        
        return scores
    
    def _von_mises_score(self, theta: float, target: float) -> float:
        """
        Score based on angular proximity to target direction.
        Uses von Mises-like function: high at target, decays with angular distance.
        """
        delta = ang_diff(theta, target)
        return math.exp(self.kappa * (math.cos(delta) - 1))
    
    def _template_seek(self, theta: float, y: Dict, agent: AgentState) -> float:
        """
        SEEK template: High score for headings toward neighbors.
        Implements approach/affiliation behavior.
        """
        if y['mean_neighbor_dir'] is None:
            return 0.5  # neutral when alone
        return self._von_mises_score(theta, y['mean_neighbor_dir'])
    
    def _template_avoid(self, theta: float, y: Dict, agent: AgentState) -> float:
        """
        AVOID template: High score for headings away from nearest neighbor.
        Implements personal space maintenance.
        """
        if y['nearest_dir'] is None:
            return 0.5
        # Away = opposite of toward
        away_dir = wrap_angle(y['nearest_dir'] + math.pi)
        # Boost when neighbor is very close
        proximity_boost = clip(2.0 / (y['nearest_dist'] + 0.5), 0.5, 2.0)
        return self._von_mises_score(theta, away_dir) * proximity_boost
    
    def _template_align(self, theta: float, y: Dict, agent: AgentState) -> float:
        """
        ALIGN template: High score for headings matching neighbor consensus.
        This is the KEY template for flocking coordination.
        """
        if y['mean_neighbor_heading'] is None:
            return 0.5
        return self._von_mises_score(theta, y['mean_neighbor_heading'])
    
    def _template_explore(self, theta: float, agent: AgentState) -> float:
        """
        EXPLORE template: Mild uniform preference with forward bias.
        Provides baseline behavior when other policies are inactive.
        """
        # Small bonus for continuing current heading (persistence)
        forward_bonus = 0.3 * self._von_mises_score(theta, agent.theta)
        return 0.5 + forward_bonus

# ============================================================================
# WORLD SIMULATION
# ============================================================================

class FlockWorld:
    """
    Main simulation environment for emotion-based flocking.
    """
    
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.N = cfg.get('num_agents', 20)
        self.dt = cfg.get('dt', 0.1)
        self.speed = cfg.get('speed', 1.0)
        self.boundary = cfg.get('boundary', 50.0)
        
        # Heading smoothing (persistence/inertia)
        # Higher = faster turning, Lower = more inertia
        self.heading_smooth = cfg.get('heading_smooth', 0.3)
        
        # Action selection temperature
        self.tau_action = cfg.get('tau_action', 0.25)
        
        # Initialize agents with random positions and headings
        self._init_agents()
        
        # Create subsystems
        self.need_system = NeedSystem(cfg)
        self.policy_system = PolicySystem(cfg)
        
        # History for plotting and analysis
        self.history = {
            'positions': [],    # List of [(x,y) for each agent] per timestep
            'headings': [],     # List of [theta for each agent] per timestep
            'arousal': [],      # Mean arousal per timestep
            'alignment': [],    # Flock alignment (order parameter) per timestep
            'nnd': [],          # Mean nearest neighbor distance per timestep
            'policy_mix': [],   # Mean policy probabilities per timestep
        }
        
        self.t = 0  # Current timestep
        
    def _init_agents(self):
        """Initialize agents with random positions and headings."""
        self.agents: List[AgentState] = []
        
        for i in range(self.N):
            # Random position within boundary
            x = random.uniform(-self.boundary * 0.8, self.boundary * 0.8)
            y = random.uniform(-self.boundary * 0.8, self.boundary * 0.8)
            # Random heading
            theta = random.uniform(-math.pi, math.pi)
            
            self.agents.append(AgentState(
                x=x, y=y, theta=theta, 
                speed=self.speed,
                color_id=i
            ))
    
    def observe(self, i: int) -> List[Tuple[float, float, float]]:
        """
        Get observations for agent i: (distance, bearing, heading) for all peers.
        """
        me = self.agents[i]
        peers = []
        
        for j, other in enumerate(self.agents):
            if i == j:
                continue
            dx = other.x - me.x
            dy = other.y - me.y
            r = math.hypot(dx, dy)
            phi = wrap_angle(math.atan2(dy, dx))  # bearing to other
            peers.append((r, phi, other.theta))
            
        return peers
    
    def step(self):
        """
        Execute one simulation step (A1 through A7 for all agents).
        """
        new_headings = []
        step_data = {'arousal': [], 'policy_mix': []}
        
        # ---- Phase 1: All agents compute their decisions ----
        for i in range(self.N):
            agent = self.agents[i]
            peers = self.observe(i)
            
            # A1: Categorize situation
            cat = categorize(agent, peers,
                           R_comfort=self.cfg.get('R_comfort', 5.0),
                           R_too_close=self.cfg.get('R_too_close', 1.5))
            
            # A2: Assess needs and compute affect
            needs = self.need_system.assess(cat['c'])
            affect = self.need_system.compute_affect(needs)
            step_data['arousal'].append(affect['arousal'])
            
            # A4: Get policy mix
            q, hints = self.policy_system.get_policy_mix(affect)
            step_data['policy_mix'].append(q)
            
            # A5: Score actions
            scores = self.policy_system.score_actions(q, cat['y'], agent)
            
            # A6: Select action
            probs = softmax(scores, self.tau_action)
            
            # Mostly argmax, occasional sampling for robustness
            if random.random() < 0.05:
                action_idx = random.choices(range(len(probs)), weights=probs)[0]
            else:
                action_idx = np.argmax(scores)
            
            new_headings.append(self.policy_system.headings[action_idx])
        
        # ---- Phase 2: Update all agents (synchronous) ----
        for i, agent in enumerate(self.agents):
            # Smooth heading change (adds persistence/inertia)
            target_heading = new_headings[i]
            delta = ang_diff(target_heading, agent.theta)
            agent.theta = wrap_angle(agent.theta + self.heading_smooth * delta)
            
            # A7: Execute - move forward
            agent.x += agent.speed * math.cos(agent.theta) * self.dt
            agent.y += agent.speed * math.sin(agent.theta) * self.dt
            
            # Boundary handling: soft reflection
            if abs(agent.x) > self.boundary:
                agent.x = np.sign(agent.x) * (2 * self.boundary - abs(agent.x))
                agent.theta = wrap_angle(math.pi - agent.theta)
            if abs(agent.y) > self.boundary:
                agent.y = np.sign(agent.y) * (2 * self.boundary - abs(agent.y))
                agent.theta = wrap_angle(-agent.theta)
        
        # ---- Log metrics ----
        self._log_step(step_data)
        self.t += 1
    
    def _log_step(self, step_data: Dict):
        """Record metrics for this timestep."""
        # Positions and headings
        positions = [(a.x, a.y) for a in self.agents]
        headings = [a.theta for a in self.agents]
        self.history['positions'].append(positions)
        self.history['headings'].append(headings)
        
        # Mean arousal
        self.history['arousal'].append(np.mean(step_data['arousal']))
        
        # Alignment (order parameter): magnitude of mean heading vector
        # 0 = random headings, 1 = perfect alignment
        hx = sum(math.cos(h) for h in headings) / self.N
        hy = sum(math.sin(h) for h in headings) / self.N
        self.history['alignment'].append(math.hypot(hx, hy))
        
        # Mean nearest neighbor distance
        nnd_list = []
        for i, (xi, yi) in enumerate(positions):
            min_d = min(math.hypot(xj - xi, yj - yi) 
                       for j, (xj, yj) in enumerate(positions) if i != j)
            nnd_list.append(min_d)
        self.history['nnd'].append(np.mean(nnd_list))
        
        # Mean policy mix
        mean_q = np.mean(step_data['policy_mix'], axis=0)
        self.history['policy_mix'].append(mean_q)
    
    def run(self, steps: int, verbose: bool = True):
        """Run simulation for given number of steps."""
        for t in range(steps):
            self.step()
            if verbose and t % 100 == 0:
                print(f"Step {t:4d}: alignment={self.history['alignment'][-1]:.2f}, "
                      f"nnd={self.history['nnd'][-1]:.2f}, "
                      f"arousal={self.history['arousal'][-1]:.2f}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def plot_trajectories_by_time(self, save_path: str = None):
        """
        Plot trajectories colored by TIME (early=blue, late=yellow).
        This shows the flow of movement clearly.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        T = len(self.history['positions'])
        
        for i in range(self.N):
            xs = [self.history['positions'][t][i][0] for t in range(T)]
            ys = [self.history['positions'][t][i][1] for t in range(T)]
            
            # Create line segments colored by time
            points = np.array([xs, ys]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Color by time
            from matplotlib.collections import LineCollection
            norm = Normalize(vmin=0, vmax=T)
            lc = LineCollection(segments, cmap='viridis', norm=norm, alpha=0.6, linewidth=1)
            lc.set_array(np.arange(T))
            ax.add_collection(lc)
        
        # Final positions with heading arrows
        for i, agent in enumerate(self.agents):
            ax.scatter(agent.x, agent.y, c='red', s=50, zorder=5)
            dx = 3 * math.cos(agent.theta)
            dy = 3 * math.sin(agent.theta)
            ax.arrow(agent.x, agent.y, dx, dy, 
                    head_width=1.5, head_length=0.8, fc='red', ec='red', zorder=6)
        
        ax.set_xlim(-self.boundary * 1.1, self.boundary * 1.1)
        ax.set_ylim(-self.boundary * 1.1, self.boundary * 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Trajectories (blue=start, yellow=end, arrows=final heading)\n{T} steps, {self.N} agents')
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=T))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Time step')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved {save_path}")
        plt.close()
    
    def plot_snapshots(self, timesteps: List[int], save_path: str = None):
        """
        Plot flock configuration at specific timesteps.
        Shows positions and headings as arrows.
        """
        n_plots = len(timesteps)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, t in enumerate(timesteps):
            if t >= len(self.history['positions']):
                continue
                
            ax = axes[idx]
            positions = self.history['positions'][t]
            headings = self.history['headings'][t]
            
            # Plot each agent as arrow
            for i, ((x, y), h) in enumerate(zip(positions, headings)):
                dx = 2 * math.cos(h)
                dy = 2 * math.sin(h)
                ax.arrow(x, y, dx, dy, head_width=1.2, head_length=0.6,
                        fc=plt.cm.tab20(i % 20), ec='black', linewidth=0.5)
            
            ax.set_xlim(-self.boundary, self.boundary)
            ax.set_ylim(-self.boundary, self.boundary)
            ax.set_aspect('equal')
            ax.set_title(f't={t}, align={self.history["alignment"][t]:.2f}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(timesteps), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Flock Snapshots')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved {save_path}")
        plt.close()
    
    def plot_metrics(self, save_path: str = None):
        """Plot time series of flock metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        T = len(self.history['alignment'])
        
        # Alignment
        ax = axes[0, 0]
        ax.plot(self.history['alignment'], 'b-', linewidth=1.5)
        ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Good alignment')
        ax.set_xlabel('Step')
        ax.set_ylabel('Alignment')
        ax.set_ylim(0, 1)
        ax.set_title('Flock Alignment (Order Parameter)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cohesion (NND)
        ax = axes[0, 1]
        ax.plot(self.history['nnd'], 'orange', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean NND')
        ax.set_title('Cohesion (Nearest Neighbor Distance)')
        ax.grid(True, alpha=0.3)
        
        # Arousal
        ax = axes[1, 0]
        ax.plot(self.history['arousal'], 'purple', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Arousal')
        ax.set_title('Group Arousal')
        ax.grid(True, alpha=0.3)
        
        # Policy mix over time
        ax = axes[1, 1]
        policy_mix = np.array(self.history['policy_mix'])
        labels = ['Seek', 'Avoid', 'Align', 'Explore']
        colors = ['green', 'red', 'blue', 'gray']
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(policy_mix[:, i], color=color, label=label, linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Policy Weight')
        ax.set_title('Mean Policy Mix')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved {save_path}")
        plt.close()
    
    def create_animation(self, save_path: str = 'flock_animation.mp4', 
                         fps: int = 30, skip: int = 2):
        """
        Create MP4 animation of the flock.
        
        Args:
            save_path: Output file path
            fps: Frames per second
            skip: Only render every `skip` timesteps (for speed)
        """
        print(f"Creating animation ({len(self.history['positions'])//skip} frames)...")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-self.boundary * 1.1, self.boundary * 1.1)
        ax.set_ylim(-self.boundary * 1.1, self.boundary * 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        frames = list(range(0, len(self.history['positions']), skip))
        
        # Initialize with first frame data
        t0 = frames[0]
        positions0 = self.history['positions'][t0]
        headings0 = self.history['headings'][t0]
        xs0 = [p[0] for p in positions0]
        ys0 = [p[1] for p in positions0]
        us0 = [math.cos(h) for h in headings0]
        vs0 = [math.sin(h) for h in headings0]
        
        quiver = ax.quiver(xs0, ys0, us0, vs0, scale=25, width=0.008, 
                          color=plt.cm.tab20(np.arange(self.N) % 20))
        title = ax.set_title(f't=0')
        
        def update(frame_idx):
            t = frames[frame_idx]
            positions = self.history['positions'][t]
            headings = self.history['headings'][t]
            
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            us = [math.cos(h) for h in headings]
            vs = [math.sin(h) for h in headings]
            
            # Update quiver
            quiver.set_offsets(np.column_stack([xs, ys]))
            quiver.set_UVC(us, vs)
            
            title.set_text(f't={t}  alignment={self.history["alignment"][t]:.2f}  '
                          f'nnd={self.history["nnd"][t]:.1f}')
            return quiver, title
        
        anim = FuncAnimation(fig, update, frames=len(frames), 
                            interval=1000//fps, blit=False)
        
        writer = FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(save_path, writer=writer)
        plt.close()
        print(f"Saved animation to {save_path}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Configuration
    cfg = {
        # Population
        'num_agents': 25,
        'dt': 0.15,
        'speed': 0.85,
        'boundary': 40.0,
        
        # Spatial parameters
        'R_comfort': 7.0,       # comfortable neighbor distance
        'R_too_close': 2.0,     # personal space radius
        
        # Need targets - key for behavior!
        'n_affiliation_target': 0.4,   # want moderate density
        'n_safety_target': 0.1,        # want low crowding
        'n_motion_target': 0.9,        # MUST keep moving
        'n_coherence_target': 0.85,    # STRONG desire for alignment
        
        # Drive sensitivity [affiliation, safety, motion, coherence]
        # Higher coherence sensitivity = stronger alignment drive
        'drive_alpha': [2.0, 3.0, 2.5, 4.0],  # coherence most sensitive
        'base_arousal': 0.4,
        
        # Policy system
        'n_headings': 36,
        'kappa': 6.0,
        'alpha_need': 0.70,  # more weight on structured need mapping
        'alpha_aff': 0.30,
        'tau_high': 0.4,     # lower = more decisive even at low arousal
        'tau_low': 0.10,
        
        # Movement dynamics
        'heading_smooth': 0.20,  # Lower = more inertia, smoother turning
        'tau_action': 0.15,      # Lower = more decisive action selection
        
        # Need-to-policy mapping H_n - TUNED for flocking
        'H_n': [
            [-1.0,  0.0,  0.3, -0.2],  # Seek ← lonely
            [ 0.0, -1.5,  0.0,  0.0],  # Avoid ← crowded (unchanged)
            [-0.2,  0.0,  0.2, -1.8],  # Align ← misaligned (STRONG!)
            [ 0.1,  0.1, -0.6,  0.0],  # Explore ← only when motion low, NOT from coherence
        ],
    }
    
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 60)
    print("EMOTION-BASED FLOCKING")
    print("Spontaneous flock formation from random initial conditions")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Agents: {cfg['num_agents']}")
    print(f"  Need targets: affil={cfg['n_affiliation_target']}, "
          f"safety={cfg['n_safety_target']}, motion={cfg['n_motion_target']}, "
          f"coherence={cfg['n_coherence_target']}")
    print(f"  Heading smoothing (inertia): {cfg['heading_smooth']}")
    print()
    
    # Create and run simulation
    world = FlockWorld(cfg)
    world.run(1000, verbose=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")

    import os
    os.makedirs('outputs', exist_ok=True)
    
    world.plot_trajectories_by_time('outputs/flock_trajectories.png')
    world.plot_snapshots([0, 100, 300, 500, 700, 900], 'outputs/flock_snapshots.png')
    world.plot_metrics('outputs/flock_metrics.png')
    
    # Create animation
    world.create_animation('outputs/flock_animation.mp4', fps=30, skip=2)
    
    print("\nDone!")