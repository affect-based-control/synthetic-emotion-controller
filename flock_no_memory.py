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

IMPORTANT NOTE ON SCOPE:
    This demonstration does NOT implement the full emotion-like control 
    architecture as defined in the paper. Specifically, episodic memory 
    (A3, A7, A8) is entirely disabled. The paper defines emotion-like control
    as requiring BOTH need-based appraisal AND episodic affective memory
    working together (dual-source architecture).
    
    This demo can be viewed as a test of GRACEFUL DEGRADATION: does the
    architecture remain functional when memory is temporarily or permanently
    unavailable? The successful flocking behavior demonstrates that the
    need-based pathway alone provides useful reactive control, even without
    the anticipatory guidance that episodic memory would provide.
    
    For the complete A1-A8 implementation with memory, see the companion
    demo: affect_memory_demo.py

SIMPLIFICATIONS IN THIS BASELINE:
    - Speed is constant (not a control variable). Agents can only adjust
      heading. This means sub-flocks cannot actively merge by speeding up
      or slowing down—merging only occurs through boundary reflections or
      heading convergence. A full implementation might include speed as a
      second control variable.
    - τ_2(a) is held constant. The full architecture allows action-selection
      temperature to vary with arousal.
    - Episodic memory is disabled (A3, A7, A8 skipped).

Key Insight:
    Heading coherence is treated as a NEED, not just a mechanical rule.
    When an agent's heading differs from neighbors, it experiences discomfort
    ("out of sync"), which drives the Align policy. This is biologically
    plausible—social animals experience discomfort when not coordinating
    with the group.

Needs System (4 needs):
    - Affiliation: Desire to be near others (prevents isolation)
    - Safety: Desire for personal space (prevents collision)  
    - Motion: Desire to keep moving (flying birds can't hover)
    - Coherence: Desire to align with neighbors (social coordination)

Policies:
    - Seek: Move toward neighbors (activated by high affiliation drive)
    - Avoid: Move away from neighbors too close (activated by high safety drive)
    - Align: Match neighbor headings (activated by high coherence drive)
    - Cruise: Maintain current heading (activated by high motion drive)

Algorithm Mapping to Paper (A1-A8):
    A1 (Categorize): categorize() → c_t, y_t
        - c_t = [c_density, c_crowding, c_coherence, c_motion]
        - y_t = situation parameters for policy instantiation
    A2 (Need Appraisal): NeedSystem.assess() and compute_affect() → n_t, z_t
        - Drives (SI convention): d_i = tanh(α_i × (n^◇_i - n_i))
        - Positive drive = deficit (want more), Negative drive = surplus
        - Valence (conventional): v_i = -d_i
        - Positive valence = feeling good, Negative valence = feeling bad
        - Affect: z_t = [v_t, m_t, a_t, d_t]
    A3 (Episodic Retrieval): SKIPPED — no memory in this baseline
        - In full implementation: z^mem_t, h^mem_t ← Retrieve_M(c_t)
    A4 (Policy Mix): PolicySystem.get_policy_mix() → q_t(π), h_t
        - h_t(π) = α_need × h^need_t(π) + α_aff × h^aff_t(π)
        - q_t(π) = softmax(h_t / τ_1(a_t))
    A5 (Policy Instantiation): PolicySystem.score_actions() → s_t(u)
        - s_t(u) = Σ_π q_t(π) × s̃_π(u)
        - Templates s̃_π(u) are innate (depend on y_t, not affect)
        - All templates return scores in [0, 1]
        - Select u_t from s_t
        - NOTE: In this baseline, τ_2(a) is held constant for simplicity.
          The full architecture allows action-selection temperature to
          vary with arousal, but here we use a fixed τ_2.
    A6 (Action Execution): Execute u_t, observe outcome
    A7 (Post-action Reappraisal): SKIPPED — no memory to update
        - In full implementation: z*_t, succ*_t ← reappraise(...)
    A8 (Episode Storage): SKIPPED — no memory in this baseline
        - In full implementation: e_t ← (c_t, z_t, h_t, z*_t, succ*_t); M ← M ∪ {e_t}

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
    """
    State of a single flocking agent.
    
    NOTE: Speed is constant (not a control variable). Agents can only
    adjust heading. This simplifies the action space but means sub-flocks
    cannot actively merge by speeding up or slowing down—merging only
    occurs through boundary reflections or heading convergence.
    """
    x: float          # position x
    y: float          # position y  
    theta: float      # heading (radians)
    speed: float      # current speed (constant, not controlled)
    
    # For visualization
    color_id: int = 0  # agent identifier for coloring

# ============================================================================
# A1: CATEGORIZATION
# ============================================================================

def categorize(agent: AgentState, peers: List[Tuple[float, float, float]], 
               R_comfort: float = 5.0, R_too_close: float = 1.5) -> Dict:
    """
    A1: Classify the current situation into category vector c_t and parameters y_t.
    
    This is the "perceptual preprocessor" that maps raw observations x_t
    to a compact situational description (c_t, y_t).
    
    Category vector c_t = [c_density, c_crowding, c_coherence, c_motion]:
        - c_density: How many neighbors within comfortable range [0,1]
        - c_crowding: Personal space violation level [0,1] (high = too close)
        - c_coherence: Agent-to-neighbor heading alignment [0,1] (1=aligned, 0=opposite)
        - c_motion: Current movement level [0,1]
    
    Args:
        agent: Current agent state
        peers: List of (distance, bearing, heading) for each neighbor
        R_comfort: Comfortable interaction distance
        R_too_close: Personal space violation distance
        
    Returns:
        Dictionary with:
        - 'c_t': Category vector [c_density, c_crowding, c_coherence, c_motion]
        - 'y_t': Parameters for policy instantiation
    """
    # Handle empty neighborhood
    if len(peers) == 0:
        return {
            'c_t': np.array([0.0, 0.0, 1.0, agent.speed]),  # c_coherence=1.0 when alone (no misalignment)
            'y_t': {
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
    
    # ---- Category: Crowding (c_crowding) ----
    # Is anyone too close? (violating personal space)
    # High value = personal space violated, safety need threatened
    min_dist = min(distances)
    if min_dist < R_too_close:
        c_crowding = clip(1.0 - min_dist / R_too_close, 0, 1)
    else:
        c_crowding = 0.0
    
    # ---- Category: Heading Coherence (c_coherence) ----
    # How well is MY heading aligned with NEIGHBORS' mean heading?
    # This measures agent-to-neighbor mismatch, not overall group alignment.
    # High value = I am aligned with neighbors, Low value = I am misaligned
    nearby_mask = [r < R_comfort * 1.5 for (r, phi, h) in peers]
    nearby_headings = [h for h, m in zip(headings, nearby_mask) if m]
    
    if len(nearby_headings) > 0:
        # Compute mean heading of neighbors (excluding self)
        mean_neighbor_heading, _ = circular_mean(nearby_headings)
        # Measure MY alignment with that mean
        delta = ang_diff(agent.theta, mean_neighbor_heading)
        c_coherence = (1.0 + math.cos(delta)) / 2.0  # 1 = aligned, 0 = opposite
    else:
        c_coherence = 1.0  # No nearby neighbors = no misalignment to feel
    
    # ---- Category: Motion (c_motion) ----
    # Am I currently moving?
    c_motion = clip(agent.speed, 0, 1)
    
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
        'c_t': np.array([c_density, c_crowding, c_coherence, c_motion]),
        'y_t': {
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
    
    Each need has a target level n^◇. Deviation from target creates DRIVES,
    which generate AFFECT (valence, magnitude, arousal).
    
    Drive convention (SI-compatible):
        d_i = tanh(α_i × (n^◇_i - n_i))
        
        Positive drive = deficit (I want more / current level below target)
        Negative drive = surplus (I have too much / current level above target)
    
    Valence convention (standard psychological):
        v_i = -d_i
        
        Positive valence = feeling good (need satisfied)
        Negative valence = feeling bad (need unsatisfied)
    
    All targets are intuitive: "I want [target] level of [need]"
        - n_affiliation_target = 0.4 → "I want moderate company"
        - n_safety_target = 0.9 → "I want HIGH safety (personal space)"
        - n_motion_target = 0.9 → "I want HIGH movement"
        - n_coherence_target = 0.85 → "I want HIGH alignment"
    """
    
    def __init__(self, cfg: Dict):
        # Need targets n^◇ (desired levels - all intuitive)
        self.n_target = np.array([
            cfg.get('n_affiliation_target', 0.4),   # want moderate density
            cfg.get('n_safety_target', 0.9),        # want HIGH safety (personal space)
            cfg.get('n_motion_target', 0.9),        # want HIGH movement
            cfg.get('n_coherence_target', 0.85),    # want HIGH alignment
        ])
        
        # Drive sensitivity α (how strongly deviation affects drives)
        # Higher = more emotional response to need deviation
        self.alpha = np.array(cfg.get('drive_alpha', [2.0, 3.0, 2.5, 4.0]))
        
        # Base arousal (background activation level)
        self.base_arousal = cfg.get('base_arousal', 0.4)
        
    def assess(self, c_t: np.ndarray) -> np.ndarray:
        """
        Map category vector c_t to need levels n_t.
        
        c_t = [c_density, c_crowding, c_coherence, c_motion]
        n_t = [n_affiliation, n_safety, n_motion, n_coherence]
        
        Note: n_safety = 1 - c_crowding (high safety when not crowded)
        """
        n_affiliation = c_t[0]        # density → affiliation level
        n_safety = 1.0 - c_t[1]       # NOT crowded → safety level (inverted)
        n_motion = c_t[3]             # moving → motion level
        n_coherence = c_t[2]          # aligned → coherence level
        
        return np.array([n_affiliation, n_safety, n_motion, n_coherence])
    
    def compute_affect(self, n_t: np.ndarray) -> Dict:
        """
        Compute affect from need levels.
        
        This implements:
            d_i = tanh(α_i × (n^◇_i - n_i))   [drives, SI convention]
            v_i = -d_i                         [valence, standard convention]
            
        Where:
        - d_i is the drive for need i:
            Positive drive = deficit (below target) = "I want more"
            Negative drive = surplus (above target) = "I have too much"
        - v_i is the valence for need i:
            Positive valence = feeling good (need satisfied)
            Negative valence = feeling bad (need unsatisfied)
        
        Returns affect dictionary with z_t = [v_t, m_t, a_t, d_t]
        """
        # Drives: signed deviation from target (SI convention: target - actual)
        raw_deviation = self.n_target - n_t
        d_t = np.tanh(self.alpha * raw_deviation)
        
        # Valence v_t: emotional tone (standard convention)
        # Positive valence = feeling good (need met, no deficit)
        # Negative valence = feeling bad (need unmet, deficit present)
        v_t = -d_t
        
        # Magnitude m_t: emotional intensity (always positive)
        m_t = np.abs(d_t)
        
        # Arousal a_t: overall activation level
        # Increases with absolute drive magnitude (distance from target in either direction)
        mean_magnitude = np.mean(m_t)
        a_t = clip(self.base_arousal + 0.4 * mean_magnitude, 0, 1)
        
        return {
            'v_t': v_t,           # [4] signed valence (positive = good, negative = bad)
            'm_t': m_t,           # [4] unsigned magnitude  
            'a_t': a_t,           # scalar arousal
            'd_t': d_t,           # [4] signed drives
            'n_t': n_t,           # [4] need levels
            # Full affect vector z_t for storage (SI notation)
            'z_t': np.concatenate([v_t, m_t, [a_t], d_t])
        }

# ============================================================================
# A4-A5: POLICY SYSTEM
# ============================================================================

class PolicySystem:
    """
    A4: Compute policy mix from affect.
    A5: Instantiate policies into action scores.
    
    Policies for flocking:
    - Seek: Move toward neighbors (activated by high affiliation drive)
    - Avoid: Move away from neighbors too close (activated by high safety drive)
    - Align: Match neighbor headings (activated by high coherence drive)
    - Cruise: Maintain current heading (activated by high motion drive)
    
    Each policy has a TEMPLATE that scores actions based on current situation.
    Templates are INNATE: they depend on situation parameters y_t, not on affect.
    All templates return scores in [0, 1].
    
    The policy MIX q_t(π) determines how much each template contributes.
    
    With SI drive convention (positive = deficit = "want more"):
    - High affiliation drive (lonely) → Seek
    - High safety drive (crowded) → Avoid  
    - High coherence drive (misaligned) → Align
    - High motion drive (not moving enough) → Cruise
    
    All H_n entries are POSITIVE for the primary policy-drive mappings.
    
    With standard valence convention (positive = good, negative = bad):
    - Negative affiliation valence (lonely, feels bad) → Seek
    - Negative safety valence (crowded, feels bad) → Avoid
    - Negative coherence valence (misaligned, feels bad) → Align
    - Negative motion valence (stuck, feels bad) → Cruise
    
    H_aff entries are NEGATIVE for primary policy-valence mappings
    (negative valence activates corrective policy).
    
    NOTE on τ_2: In this baseline, τ_2(a) is held constant for simplicity.
    The full architecture allows action-selection temperature to vary with
    arousal (higher arousal → lower temperature → more decisive actions),
    but here we use a fixed τ_2 value.
    """
    
    def __init__(self, cfg: Dict):
        self.policies = ['Seek', 'Avoid', 'Align', 'Cruise']
        self.n_policies = len(self.policies)
        
        # Action space: discrete headings
        self.n_headings = cfg.get('n_headings', 36)
        self.headings = np.linspace(-math.pi, math.pi, self.n_headings, endpoint=False)
        
        # Von Mises concentration for action scoring
        self.kappa = cfg.get('kappa', 5.0)
        
        # Spatial parameter for Avoid template
        self.R_too_close = cfg.get('R_too_close', 2.0)
        
        # ---- H_n: Drive-to-Policy Mapping (SI convention) ----
        # Maps drives d_t = [affiliation, safety, motion, coherence] to policy hints
        # 
        # With SI convention: positive drive = deficit = "want more"
        # Positive H_n entry: high drive → high policy activation
        #
        # All primary mappings are positive and intuitive:
        #   High affiliation drive (lonely) → Seek
        #   High safety drive (crowded) → Avoid
        #   High coherence drive (misaligned) → Align
        #   High motion drive (stuck) → Cruise
        self.H_n = np.array(cfg.get('H_n', [
            # Columns: [affiliation, safety, motion, coherence]
            # Seek: HIGH affiliation drive (lonely, want company)
            [ 1.0,  0.0, -0.3,  0.2],
            # Avoid: HIGH safety drive (crowded, want space)
            [ 0.0,  1.5,  0.0,  0.0],
            # Align: HIGH coherence drive (misaligned, want sync)
            [ 0.2,  0.0, -0.2,  1.8],
            # Cruise: HIGH motion drive (stuck, want movement)
            [-0.1, -0.1,  0.6,  0.0],
        ]))
        
        # ---- H_aff: Affect-to-Policy Mapping (standard valence convention) ----
        # Maps [v_0, v_1, v_2, v_3, a_t] to policy hints
        # 
        # With standard valence: positive = good, negative = bad
        # NEGATIVE H_aff entries for primary mappings:
        #   Negative valence (feeling bad) → activate corrective policy
        #
        # Example: lonely → negative affiliation valence → 
        #          negative × negative H_aff entry → positive Seek activation
        self.H_aff = np.array(cfg.get('H_aff', [
            # Columns: [v_affil, v_safety, v_motion, v_coherence, arousal]
            # Seek: NEGATIVE affiliation valence (lonely feels bad) + arousal boost
            [-0.3,  0.0,  0.0,  0.0,  0.2],
            # Avoid: NEGATIVE safety valence (crowded feels bad) + strong arousal boost
            [ 0.0, -0.4,  0.0,  0.0,  0.3],
            # Align: NEGATIVE coherence valence (misaligned feels bad) + arousal
            [ 0.0,  0.0,  0.0, -0.5,  0.1],
            # Cruise: low arousal (cruise when calm)
            [-0.025, -0.025, -0.025, -0.025, -0.2],
        ]))
        
        # ---- Fusion Weights (α_need, α_aff in SI) ----
        self.alpha_need = cfg.get('alpha_need', 0.70)  # weight for need-based hints
        self.alpha_aff = cfg.get('alpha_aff', 0.30)    # weight for affect-based hints
        
        # ---- Temperature Schedule τ_1(a_t) ----
        # Maps arousal to policy selection temperature
        # High arousal → low temperature → decisive (sharp distribution)
        # Low arousal → high temperature → exploratory (flat distribution)
        self.tau_1_high = cfg.get('tau_1_high', 0.4)  # temperature at low arousal
        self.tau_1_low = cfg.get('tau_1_low', 0.10)   # temperature at high arousal
        
    def compute_hints_from_needs(self, d_t: np.ndarray) -> np.ndarray:
        """
        Compute policy hints from drives using H_n matrix.
        
        h^need_t(π) = [H_n × d_t]_π
        
        With SI convention and positive H_n entries:
        - Positive drive (deficit) → positive hint → policy activated
        """
        h_need_t = self.H_n @ d_t
        # Normalize to [-1, 1] for consistent scale
        max_abs = np.abs(h_need_t).max()
        if max_abs > 1e-6:
            h_need_t = h_need_t / max_abs
        return h_need_t
    
    def compute_hints_from_affect(self, affect: Dict) -> np.ndarray:
        """
        Compute policy hints from affect state using H_aff matrix.
        
        h^aff_t(π) = [H_aff × [v_t; a_t]]_π
        
        With standard valence convention (positive = good, negative = bad)
        and negative H_aff entries for primary mappings:
        - Negative valence × negative H_aff = positive policy activation
        
        Example: lonely → v_affil < 0 → (-0.3) × v_affil > 0 → Seek activated
        """
        v_t = affect['v_t']
        a_t = affect['a_t']
        
        # Construct input vector [v_0, v_1, v_2, v_3, a]
        aff_input = np.concatenate([v_t, [a_t]])
        
        h_aff_t = self.H_aff @ aff_input
        
        max_abs = np.abs(h_aff_t).max()
        if max_abs > 1e-6:
            h_aff_t = h_aff_t / max_abs
        return h_aff_t
    
    def get_policy_mix(self, affect: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        A4: Fuse hints and compute policy probabilities q_t(π).
        
        h_t(π) = α_need × h^need_t(π) + α_aff × h^aff_t(π)
        q_t(π) = softmax(h_t(π) / τ_1(a_t))
        
        Returns:
            q_t: Policy probabilities [n_policies]
            h_t: Fused hints [n_policies] (for logging)
        """
        d_t = affect['d_t']
        a_t = affect['a_t']
        
        # Compute hints from both sources
        h_need_t = self.compute_hints_from_needs(d_t)
        h_aff_t = self.compute_hints_from_affect(affect)
        
        # Fuse hints (weighted combination, SI Eq. 11)
        h_t = self.alpha_need * h_need_t + self.alpha_aff * h_aff_t
        
        # Policy temperature τ_1(a_t): decreases with arousal
        tau_1 = self.tau_1_high - (self.tau_1_high - self.tau_1_low) * a_t
        
        # Convert to probabilities (SI Eq. 12)
        q_t = softmax(h_t, tau_1)
        
        return q_t, h_t
    
    def score_actions(self, q_t: np.ndarray, y_t: Dict, agent: AgentState) -> np.ndarray:
        """
        A5: Score each action using policy-weighted templates.
        
        s_t(u) = Σ_π q_t(π) × s̃_π(u)
        
        Each policy template s̃_π(u) scores how well action u
        implements policy π in the current situation.
        
        Templates are INNATE: they depend on y_t (situation), not affect.
        All templates return scores in [0, 1].
        
        NOTE: In this baseline, τ_2(a) is held constant for simplicity.
        The full architecture allows action-selection temperature to vary
        with arousal, but here we use a fixed τ_2 value.
        """
        s_t = np.zeros(self.n_headings)
        
        for i, theta in enumerate(self.headings):
            # Get template scores s̃_π(u) for this heading
            s_seek = self._template_seek(theta, y_t, agent)
            s_avoid = self._template_avoid(theta, y_t, agent)
            s_align = self._template_align(theta, y_t, agent)
            s_cruise = self._template_cruise(theta, agent)
            
            # Policy-weighted combination
            s_t[i] = (q_t[0] * s_seek + 
                      q_t[1] * s_avoid + 
                      q_t[2] * s_align + 
                      q_t[3] * s_cruise)
        
        return s_t
    
    def _von_mises_score(self, theta: float, target: float) -> float:
        """
        Score based on angular proximity to target direction.
        Uses von Mises-like function: high at target, decays with angular distance.
        Returns score in [0, 1].
        """
        delta = ang_diff(theta, target)
        return math.exp(self.kappa * (math.cos(delta) - 1))
    
    def _template_seek(self, theta: float, y_t: Dict, agent: AgentState) -> float:
        """
        SEEK template s̃_Seek(u): High score for headings toward neighbors.
        Implements approach/affiliation behavior.
        Returns score in [0, 1].
        """
        if y_t['mean_neighbor_dir'] is None:
            return 0.5  # neutral when alone
        return self._von_mises_score(theta, y_t['mean_neighbor_dir'])
    
    def _template_avoid(self, theta: float, y_t: Dict, agent: AgentState) -> float:
        """
        AVOID template s̃_Avoid(u): High score for headings away from nearest neighbor.
        Implements personal space maintenance / safety behavior.
        
        This is an INNATE template: it depends on situation parameters y_t,
        not on affect. The intensity scaling by proximity is part of the
        innate mapping from situation to action preference.
        
        Returns score in [0, 1].
        """
        if y_t['nearest_dir'] is None:
            return 0.5
        # Away = opposite of toward
        away_dir = wrap_angle(y_t['nearest_dir'] + math.pi)
        # Situation-dependent intensity (innate, based on y_t not affect)
        # Closer neighbors → higher score, normalized to [0.5, 1.0]
        proximity_factor = clip(1.0 - y_t['nearest_dist'] / (2 * self.R_too_close), 0.5, 1.0)
        return self._von_mises_score(theta, away_dir) * proximity_factor
    
    def _template_align(self, theta: float, y_t: Dict, agent: AgentState) -> float:
        """
        ALIGN template s̃_Align(u): High score for headings matching neighbor consensus.
        This is the KEY template for flocking coordination.
        Returns score in [0, 1].
        """
        if y_t['mean_neighbor_heading'] is None:
            return 0.5
        return self._von_mises_score(theta, y_t['mean_neighbor_heading'])
    
    def _template_cruise(self, theta: float, agent: AgentState) -> float:
        """
        CRUISE template s̃_Cruise(u): High score for maintaining current heading.
        Provides heading persistence / stability when other policies are inactive.
        Returns score in [0, 1].
        """
        return self._von_mises_score(theta, agent.theta)

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
        
        # Action selection temperature τ_2 (constant in this baseline)
        # NOTE: In the full architecture, τ_2(a) varies with arousal.
        # Here we use a fixed value for simplicity.
        self.tau_2 = cfg.get('tau_2', 0.25)
        
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
            'valence': [],      # Mean valence per timestep
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
        Execute one simulation step following A1-A8 structure.
        
        Note: A3, A7, A8 are SKIPPED in this no-memory baseline.
        This tests graceful degradation when episodic memory is unavailable.
        """
        new_headings = []
        step_data = {'arousal': [], 'valence': [], 'policy_mix': []}
        
        # ---- Phase 1: All agents compute their decisions ----
        for i in range(self.N):
            agent = self.agents[i]
            peers = self.observe(i)
            
            # ============================================================
            # A1: Categorize situation → c_t, y_t
            # ============================================================
            cat = categorize(agent, peers,
                           R_comfort=self.cfg.get('R_comfort', 5.0),
                           R_too_close=self.cfg.get('R_too_close', 1.5))
            c_t = cat['c_t']
            y_t = cat['y_t']
            
            # ============================================================
            # A2: Assess needs and compute affect → n_t, z_t
            # ============================================================
            n_t = self.need_system.assess(c_t)
            affect = self.need_system.compute_affect(n_t)
            step_data['arousal'].append(affect['a_t'])
            step_data['valence'].append(np.mean(affect['v_t']))  # mean valence
            
            # ============================================================
            # A3: Episodic retrieval — SKIPPED (no memory in this baseline)
            #
            # In full implementation this would be:
            #     z^mem_t, h^mem_t ← Retrieve_M(c_t)
            #
            # Without memory, we rely solely on need-based and affect-based
            # signals. This demonstrates graceful degradation: the system
            # remains functional but lacks anticipatory guidance from past
            # episodes.
            # ============================================================
            
            # ============================================================
            # A4: Get policy mix → q_t(π), h_t
            # 
            # Note: Without A3, h_t uses only h^need_t and h^aff_t,
            # missing the h^mem_t contribution from episodic memory.
            # ============================================================
            q_t, h_t = self.policy_system.get_policy_mix(affect)
            step_data['policy_mix'].append(q_t)
            
            # ============================================================
            # A5: Policy instantiation — score actions → s_t(u)
            #     Then select action u_t
            #
            # NOTE: In this baseline, τ_2(a) is held constant for simplicity.
            # The full architecture allows action-selection temperature to
            # vary with arousal, but here we use a fixed τ_2 value.
            # ============================================================
            s_t = self.policy_system.score_actions(q_t, y_t, agent)
            probs = softmax(s_t, self.tau_2)
            
            # Mostly argmax, occasional sampling for robustness
            if random.random() < 0.05:
                action_idx = random.choices(range(len(probs)), weights=probs)[0]
            else:
                action_idx = np.argmax(s_t)
            
            new_headings.append(self.policy_system.headings[action_idx])
        
        # ---- Phase 2: Update all agents (synchronous) ----
        for i, agent in enumerate(self.agents):
            # Smooth heading change (adds persistence/inertia)
            target_heading = new_headings[i]
            delta = ang_diff(target_heading, agent.theta)
            agent.theta = wrap_angle(agent.theta + self.heading_smooth * delta)
            
            # ============================================================
            # A6: Action execution — perform action and observe outcome
            # ============================================================
            agent.x += agent.speed * math.cos(agent.theta) * self.dt
            agent.y += agent.speed * math.sin(agent.theta) * self.dt
            
            # Boundary handling: soft reflection
            if abs(agent.x) > self.boundary:
                agent.x = np.sign(agent.x) * (2 * self.boundary - abs(agent.x))
                agent.theta = wrap_angle(math.pi - agent.theta)
            if abs(agent.y) > self.boundary:
                agent.y = np.sign(agent.y) * (2 * self.boundary - abs(agent.y))
                agent.theta = wrap_angle(-agent.theta)
        
            # ============================================================
            # A7: Post-action reappraisal — SKIPPED (no memory to update)
            #
            # In full implementation this would be:
            #     z*_t, succ*_t ← reappraise(z_t, x_t, n_t, x*_t, n*_t)
            # ============================================================
            
            # ============================================================
            # A8: Episode storage — SKIPPED (no memory in this baseline)
            #
            # In full implementation this would be:
            #     e_t ← (c_t, z_t, h_t, z*_t, succ*_t)
            #     M ← M ∪ {e_t}
            # ============================================================
        
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
        
        # Mean valence
        self.history['valence'].append(np.mean(step_data['valence']))
        
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
                      f"valence={self.history['valence'][-1]:.2f}, "
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
        
        plt.suptitle('Flock Snapshots (No-Memory Baseline)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved {save_path}")
        plt.close()
    
    def plot_metrics(self, save_path: str = None):
        """Plot time series of flock metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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
        
        # Valence
        ax = axes[0, 2]
        ax.plot(self.history['valence'], 'green', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Valence')
        ax.set_title('Group Valence (+ = good, − = bad)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Arousal
        ax = axes[1, 0]
        ax.plot(self.history['arousal'], 'purple', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Arousal')
        ax.set_title('Group Arousal')
        ax.grid(True, alpha=0.3)
        
        # Valence vs Arousal scatter
        ax = axes[1, 1]
        colors = np.linspace(0, 1, T)
        sc = ax.scatter(self.history['valence'], self.history['arousal'], 
                       c=colors, cmap='viridis', alpha=0.5, s=10)
        ax.set_xlabel('Valence (+ = good)')
        ax.set_ylabel('Arousal')
        ax.set_title('Affect Space Trajectory')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Time')
        ax.grid(True, alpha=0.3)
        
        # Policy mix over time
        ax = axes[1, 2]
        policy_mix = np.array(self.history['policy_mix'])
        labels = ['Seek', 'Avoid', 'Align', 'Cruise']
        colors = ['green', 'red', 'blue', 'gray']
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(policy_mix[:, i], color=color, label=label, linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Policy Weight')
        ax.set_title('Mean Policy Mix')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Flock Metrics (No-Memory Baseline - Graceful Degradation Test)')
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
                          f'valence={self.history["valence"][t]:.2f}  '
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
        # NOTE: Speed is constant (not a control variable). Agents can only
        # adjust heading. This means sub-flocks cannot actively merge by
        # speeding up or slowing down—merging only occurs through boundary
        # reflections or heading convergence.
        'speed': 0.85,
        'boundary': 40.0,
        
        # Spatial parameters
        'R_comfort': 7.0,       # comfortable neighbor distance
        'R_too_close': 2.0,     # personal space radius
        
        # Need targets n^◇ — all intuitive: "I want [target] level of [need]"
        'n_affiliation_target': 0.4,   # want MODERATE density (not too isolated)
        'n_safety_target': 0.9,        # want HIGH safety (personal space respected)
        'n_motion_target': 0.9,        # want HIGH motion (keep moving)
        'n_coherence_target': 0.85,    # want HIGH coherence (aligned with group)
        
        # Drive sensitivity α = [affiliation, safety, motion, coherence]
        # Higher = more emotional response to deviation from target
        'drive_alpha': [2.0, 3.0, 2.5, 4.0],  # coherence most sensitive
        'base_arousal': 0.4,
        
        # Policy system
        'n_headings': 36,
        'kappa': 6.0,
        'alpha_need': 0.70,    # weight for need-based hints h^need_t
        'alpha_aff': 0.30,     # weight for affect-based hints h^aff_t
        'tau_1_high': 0.4,     # policy temperature τ_1 at low arousal
        'tau_1_low': 0.10,     # policy temperature τ_1 at high arousal
        
        # Movement dynamics
        'heading_smooth': 0.20,  # Lower = more inertia, smoother turning
        # NOTE: τ_2 is held constant in this baseline. The full architecture
        # allows τ_2(a) to vary with arousal.
        'tau_2': 0.15,           # action selection temperature τ_2 (constant)
        
        # Drive-to-policy mapping H_n (SI convention: positive drive = deficit)
        # All primary mappings are POSITIVE: high drive → policy activated
        'H_n': [
            # Columns: [affiliation, safety, motion, coherence]
            [ 1.0,  0.0, -0.3,  0.2],  # Seek ← high affiliation drive (lonely)
            [ 0.0,  1.5,  0.0,  0.0],  # Avoid ← high safety drive (crowded)
            [ 0.2,  0.0, -0.2,  1.8],  # Align ← high coherence drive (misaligned)
            [-0.1, -0.1,  0.6,  0.0],  # Cruise ← high motion drive (stuck)
        ],
        
        # Affect-to-policy mapping H_aff (standard valence: positive = good)
        # Primary mappings are NEGATIVE: negative valence (bad) → policy activated
        'H_aff': [
            # Columns: [v_affil, v_safety, v_motion, v_coherence, arousal]
            [-0.3,  0.0,  0.0,  0.0,  0.2],   # Seek ← negative affil valence (lonely feels bad)
            [ 0.0, -0.4,  0.0,  0.0,  0.3],   # Avoid ← negative safety valence (crowded feels bad)
            [ 0.0,  0.0,  0.0, -0.5,  0.1],   # Align ← negative coherence valence (misaligned feels bad)
            [-0.025, -0.025, -0.025, -0.025, -0.2],  # Cruise ← general negative valence, low arousal
        ],
    }
    
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("EMOTION-BASED FLOCKING (No-Memory Baseline)")
    print("=" * 70)
    print()
    print("NOTE: This demo tests GRACEFUL DEGRADATION of the emotion-like")
    print("      control architecture. Episodic memory (A3, A7, A8) is disabled.")
    print("      Full emotion-like control requires dual-source integration")
    print("      (needs + memory). See affect_memory_demo.py for full A1-A8.")
    print()
    print("SIMPLIFICATIONS IN THIS BASELINE:")
    print("  - Speed is constant (heading is the only control variable)")
    print("  - τ_2(a) is held constant (not arousal-dependent)")
    print("  - Episodic memory disabled (A3, A7, A8 skipped)")
    print()
    print("Valence Convention (standard psychological):")
    print("  v_i = -d_i")
    print("  Positive valence = feeling good (need satisfied)")
    print("  Negative valence = feeling bad (need unsatisfied)")
    print()
    print("Configuration:")
    print(f"  Agents: {cfg['num_agents']}")
    print(f"  Speed: {cfg['speed']} (constant, not controlled)")
    print(f"  Need targets n^◇:")
    print(f"    affiliation = {cfg['n_affiliation_target']} (want moderate company)")
    print(f"    safety      = {cfg['n_safety_target']} (want HIGH personal space)")
    print(f"    motion      = {cfg['n_motion_target']} (want HIGH movement)")
    print(f"    coherence   = {cfg['n_coherence_target']} (want HIGH alignment)")
    print(f"  Drive sensitivity α: {cfg['drive_alpha']}")
    print(f"  Drive convention: d_i = tanh(α_i × (n^◇_i - n_i)) [SI]")
    print(f"    Positive drive = deficit (want more)")
    print(f"    Negative drive = surplus (have too much)")
    print(f"  Action temperature: τ_2 = {cfg['tau_2']} (constant in this baseline)")
    print()
    
    # Create and run simulation
    world = FlockWorld(cfg)
    world.run(2000, verbose=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")

    import os
    os.makedirs('outputs', exist_ok=True)
    
    world.plot_trajectories_by_time('outputs/flock_trajectories.png')
    world.plot_snapshots([0, 400, 800, 1200, 1600, 1900], 'outputs/flock_snapshots.png')
    world.plot_metrics('outputs/flock_metrics.png')
    
    # Create animation (comment out if ffmpeg not available)
    try:
        world.create_animation('outputs/flock_animation.mp4', fps=30, skip=2)
    except Exception as e:
        print(f"Animation skipped (ffmpeg may not be available): {e}")
    
    print("\nDone!")
