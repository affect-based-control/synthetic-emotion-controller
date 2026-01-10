#!/usr/bin/env python3
"""
================================================================================
AFFECT-BASED EPISODIC MEMORY FOR ANTICIPATORY BEHAVIOR
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

Implementation of Algorithm A1-A8 demonstrating episodic affect memory
for anticipatory harm avoidance.

Scenario:
    Agents explore a 2D world with a harm zone (y > 0) that delivers
    "electric shock"—strong negative affect upon contact, with no
    perceptual warning signal beforehand.

Key Demonstration:
    - Without memory: Agents rely on immediate perception, frequently
      entering harm zone and experiencing negative affect reactively.
    - With memory: Agents recall negative affect from similar past
      situations, triggering avoidance BEFORE crossing into harm.

Key Mechanism:
    Affect comes from raw sensation (the shock at y > 0), but memory
    retrieval uses coarse categories (y-bands). An agent at y = -1
    (safe, no shock) retrieves memories from y = +1 (shocked, same band),
    providing anticipatory warning that immediate perception alone
    cannot provide.

Emotional Drives:
    - Curiosity: Intrinsic drive to explore the environment
    - Pain avoidance: Contact with harm zone causes strong negative affect
    - The tension between curiosity and remembered pain shapes behavior

Algorithm Structure:
    A1: Observe and categorize → (c, y) from x
    A2: Assess needs → z_need, h_need
    A3: Retrieve episodic memory → z_mem, h_mem
    A4: Fuse affect and hints → z, h, q(π)
    A5: Instantiate policy → action scores
    A6-A7: Execute and reappraise → z*, succ*
    A8: Store episode (only if |succ*| > threshold)

Design Notes:
    - Policy-to-action mappings (e.g., Flee → go south) are assumed innate.
      Learning WHEN to trigger policies is demonstrated; learning HOW to
      execute them would require extended temporal windows (future work).
    - Only significant episodes (|succ*| > threshold) are stored to prevent
      memory dilution from neutral experiences.
    - Single-step episodes are intertwined with innate policy-to-action:
      with single-step episodes, an agent can learn WHEN to flee but not
      WHERE fleeing should lead.
    - The agent only "knows" about harm through affect (negative valence),
      not through direct access to world coordinates. The y > 0 check
      occurs only in the sensory interface (assess_affect_from_raw_state).

Expected Results:
    - Early crossings: Similar for both conditions (learning period)
    - Late crossings: ~0 with memory vs. continued crossings without
    - Late crossing reduction: ~100%

Usage:
    python affect_memory_demo.py  
    (or copy/paste into jupyter notebook, cf. e.g. jupyter.org)
    
Outputs:
    - outputs/harm-algorithm_comparison.png: Comparison plots
    - outputs/harm-no_memory.mp4: Animation without memory
    - outputs/harm-with_memory.mp4: Animation with memory

Reference:
    "Synthetic Emotions and Consciousness: Exploring Architectural Boundaries"
    See paper Figure 1 and Supplementary Information Part I.

Repository:
    https://github.com/affect-based-control/synthetic-emotion-controller
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, FFMpegWriter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    """Clamp value to range [lo, hi]."""
    return max(lo, min(hi, x))


def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature."""
    x = np.asarray(x, dtype=float)
    tau = max(tau, 1e-6)
    x = x - np.max(x)
    e = np.exp(x / tau)
    return e / (e.sum() + 1e-12)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # --- World Parameters ---
    'num_agents': 30,
    'boundary_x': 20.0,
    'boundary_y': 10.0,
    'n_y_bands': 5,
    'dt': 0.25,
    'base_speed': 0.8,
    
    # --- Policy Space ---
    'n_policies': 2,
    
    # --- Affect Parameters (A2) ---
    'harm_intensity': 0.95,
    'curiosity_intensity': 0.3,
    'base_arousal': 0.3,
    'pain_threshold': -0.5,
    
    # --- Success Evaluation Weights (A7) ---
    'success_weight_harm': 5.0,
    'success_weight_curiosity': 1.0,
    
    # --- Affect-to-Hint Mapping (A4, line 10) ---
    'H_aff_explore_curiosity': 1.2,
    'H_aff_explore_harm': 0.3,
    'H_aff_flee': -2.5,
    
    # --- Hint Fusion Weights (A4, line 11) ---
    'alpha_need': 0.05,
    'alpha_mem': 0.85,
    'alpha_aff': 0.5,
    
    # --- Affect Fusion (A4, line 9) ---
    'alpha_z_mem': 0.9,
    
    # --- Policy Selection (A4, line 12) ---
    'tau_base': 0.5,
    'tau_arousal_factor': 0.4,
    
    # --- Episodic Memory (A3, A8) ---
    'memory_max': 100,
    'memory_K': 8,
    'tau_retrieval': 2.0,
    'alpha_tag': 0.1,
    'store_threshold': 0.4,  # Adjusted for weighted per-channel formula
    
    # --- Movement Dynamics ---
    'heading_smooth': 0.3,
    'arousal_smooth_boost': 0.6,
    
    # --- Reactive Behavior ---
    'reactive_flee_prob': 0.35,
}


# ============================================================================
# EPISODE DATA STRUCTURE (A8, line 19)
# ============================================================================

@dataclass
class Episode:
    """Episodic memory record: e_t = (t, c_t, z_t, h_t, z*_t, succ*_t)"""
    c: np.ndarray
    z: np.ndarray
    h: np.ndarray
    z_star: np.ndarray
    succ_star: float


# ============================================================================
# EPISODIC MEMORY SYSTEM (A3: lines 6-8, A8: lines 19-20)
# ============================================================================

class EpisodicMemory:
    """Episodic memory with retrieval and storage."""
    
    def __init__(self, cfg: Dict):
        self.max_size = cfg['memory_max']
        self.K = cfg['memory_K']
        self.tau_retrieval = cfg['tau_retrieval']
        self.alpha_tag = cfg['alpha_tag']
        self.n_policies = cfg['n_policies']
        self.store_threshold = cfg['store_threshold']
        self.episodes: deque = deque(maxlen=self.max_size)
    
    def store(self, episode: Episode) -> bool:
        """A8: Store episode if significant (|succ*| > threshold)."""
        if abs(episode.succ_star) >= self.store_threshold:
            self.episodes.append(episode)
            return True
        return False
    
    def retrieve(self, c_query: np.ndarray) -> Tuple[Optional[np.ndarray], 
                                                      Optional[np.ndarray], 
                                                      float]:
        """A3: Retrieve affect and hints from similar episodes."""
        if len(self.episodes) == 0:
            return None, None, 0.0
        
        c_query = np.asarray(c_query).flatten()
        c_query_norm = np.linalg.norm(c_query) + 1e-12
        
        similarities = []
        for ep in self.episodes:
            c_ep = np.asarray(ep.c).flatten()
            c_ep_norm = np.linalg.norm(c_ep) + 1e-12
            sim = np.dot(c_query, c_ep) / (c_query_norm * c_ep_norm)
            similarities.append(sim)
        
        similarities = np.array(similarities)
        K = min(self.K, len(self.episodes))
        top_indices = np.argsort(similarities)[-K:]
        top_similarities = similarities[top_indices]
        weights = softmax(top_similarities, 1.0 / self.tau_retrieval)
        
        # Line 7: z_mem
        z_mem = np.zeros_like(self.episodes[0].z)
        for idx, w in zip(top_indices, weights):
            ep = self.episodes[idx]
            z_tag = self.alpha_tag * ep.z + (1 - self.alpha_tag) * ep.z_star
            z_mem += w * z_tag
        
        # Line 8: h_mem with signed success
        h_mem = np.zeros(self.n_policies)
        denominator = 0.0
        for idx, w in zip(top_indices, weights):
            ep = self.episodes[idx]
            h_mem += w * ep.succ_star * ep.h
            denominator += w * abs(ep.succ_star)
        
        if denominator > 1e-6:
            h_mem = h_mem / denominator
        
        reliability = float(np.max(top_similarities))
        return z_mem, h_mem, reliability
    
    def __len__(self) -> int:
        return len(self.episodes)


# ============================================================================
# AGENT
# ============================================================================

class Agent:
    """Agent implementing the affect-based control loop (A1-A8).
    
    Key design: The agent only "knows" about harm through AFFECT (negative
    valence), not through direct access to world coordinates. The y > 0 
    check occurs only in the sensory interface (assess_affect_from_raw_state).
    """
    
    def __init__(self, x: float, y: float, theta: float, cfg: Dict):
        self.x = x
        self.y = y
        self.theta = theta
        self.cfg = cfg
        self.memory = EpisodicMemory(cfg)
    
    def get_category(self) -> np.ndarray:
        """A1: Categorize into coarse y-band."""
        n_bands = self.cfg['n_y_bands']
        boundary_y = self.cfg['boundary_y']
        band_size = 2 * boundary_y / n_bands
        band_index = int((self.y + boundary_y) / band_size)
        band_index = clip(band_index, 0, n_bands - 1)
        
        category = np.zeros(n_bands)
        category[band_index] = 1.0
        return category
    
    def assess_affect_from_raw_state(self, y_position: float) -> Dict:
        """A2: Compute affect from raw sensory state.
        
        This is the ONLY place where y > 0 is checked directly.
        The agent experiences harm as negative valence, not as 
        knowledge of world coordinates.
        """
        # Sensory interface: world state -> affect
        in_harm_zone = y_position > 0
        harm_valence = -self.cfg['harm_intensity'] if in_harm_zone else 0.0
        curiosity_valence = self.cfg['curiosity_intensity']
        arousal = self.cfg['base_arousal'] + 0.5 * abs(harm_valence)
        arousal = clip(arousal, 0.0, 1.0)
        z = np.array([harm_valence, curiosity_valence, arousal])
        
        # Determine if "in pain" based on affect, not coordinates
        in_pain = harm_valence < self.cfg['pain_threshold']
        
        return {
            'z': z,
            'harm_valence': harm_valence,
            'curiosity_valence': curiosity_valence,
            'arousal': arousal,
            'in_pain': in_pain  # Derived from affect, used by agent
        }
    
    def compute_hints_from_needs(self) -> np.ndarray:
        """A2, line 5: Baseline hints from needs."""
        return np.array([0.1, 0.0])
    
    def fuse_affect(self, z_need: np.ndarray, 
                    z_mem: Optional[np.ndarray],
                    reliability: float) -> np.ndarray:
        """A4, line 9: Fuse need-based and memory-based affect."""
        if z_mem is None:
            return z_need
        alpha = self.cfg['alpha_z_mem'] * reliability
        min_len = min(len(z_need), len(z_mem))
        return (1 - alpha) * z_need[:min_len] + alpha * z_mem[:min_len]
    
    def compute_hints_from_affect(self, z: np.ndarray) -> np.ndarray:
        """A4, line 10: Compute policy hints from fused affect."""
        harm_valence = z[0]
        curiosity_valence = z[1] if len(z) > 1 else self.cfg['curiosity_intensity']
        
        h_explore = (self.cfg['H_aff_explore_curiosity'] * curiosity_valence +
                     self.cfg['H_aff_explore_harm'] * harm_valence)
        h_flee = self.cfg['H_aff_flee'] * harm_valence
        
        return np.array([h_explore, h_flee])
    
    def fuse_hints(self, h_need: np.ndarray,
                   h_mem: Optional[np.ndarray],
                   h_aff: np.ndarray,
                   reliability: float) -> np.ndarray:
        """A4, line 11: Fuse hints from all sources."""
        h = self.cfg['alpha_need'] * h_need + self.cfg['alpha_aff'] * h_aff
        if h_mem is not None:
            effective_alpha_mem = self.cfg['alpha_mem'] * reliability
            h = h + effective_alpha_mem * h_mem
        return h
    
    def select_policy(self, h: np.ndarray, arousal: float, in_pain: bool) -> int:
        """A4, line 12: Select policy via softmax.
        
        Args:
            in_pain: Based on affect (harm_valence < threshold), not y > 0
        """
        # Reactive flee when experiencing pain
        if in_pain and random.random() < self.cfg['reactive_flee_prob']:
            return 1
        
        tau = self.cfg['tau_base'] - self.cfg['tau_arousal_factor'] * arousal
        tau = max(tau, 0.1)
        q = softmax(h, tau)
        
        if random.random() < 0.95:
            return int(np.argmax(q))
        else:
            return random.choices(range(len(q)), weights=q)[0]
    
    def get_action_for_policy(self, policy: int) -> float:
        """A5: Convert policy to action (heading).
    
        Flee direction (south) is innate - an evolved response toward 
        a predetermined safe region. We demonstrate the use of memory
        to generate anticipatory triggering of FLEE via stored episodes, 
        not to learn the flee direction itself (future work).
        """
        if policy == 0:  # Explore
            return random.uniform(-math.pi, math.pi)
        else:  # Flee - innate taxis toward safe region
            return -math.pi / 2 + random.gauss(0, 0.2)
    
    def execute_action(self, desired_theta: float, arousal: float) -> None:
        """A6: Execute action."""
        smooth = self.cfg['heading_smooth'] + self.cfg['arousal_smooth_boost'] * arousal
        smooth = clip(smooth, 0.1, 0.9)
        
        delta = math.atan2(math.sin(desired_theta - self.theta),
                          math.cos(desired_theta - self.theta))
        self.theta += smooth * delta
        
        self.x += self.cfg['base_speed'] * math.cos(self.theta) * self.cfg['dt']
        self.y += self.cfg['base_speed'] * math.sin(self.theta) * self.cfg['dt']
        
        bx, by = self.cfg['boundary_x'], self.cfg['boundary_y']
        self.x = clip(self.x, -bx + 0.1, bx - 0.1)
        
        if self.y > by - 0.1:
            self.y = by - 0.1
            self.theta = -abs(self.theta)
        if self.y < -by + 0.1:
            self.y = -by + 0.1
            self.theta = abs(self.theta)
    
    def compute_success_from_affect(self, z_pre: np.ndarray, z_post: np.ndarray) -> float:
        """A7: Compute success using weighted per-channel hedonic evaluation.
        
        For each affect channel i, compute normalized success:
            succ_i = (v*_i m*_i - v_i m_i) / (2 * max(|v*_i m*_i|, |v_i m_i|, ε))
        
        Then aggregate via weighted average:
            succ* = Σ_i w_i succ_i / Σ_i w_i
        """
        epsilon = 0.01
    
        weights = np.array([self.cfg['success_weight_harm'], 
                            self.cfg['success_weight_curiosity']])
    
        v_pre = z_pre[:2]
        m_pre = np.abs(z_pre[:2])
        v_post = z_post[:2]
        m_post = np.abs(z_post[:2])
    
        vm_pre = v_pre * m_pre
        vm_post = v_post * m_post
    
        # Per-channel success with factor of 2 for correct bounds
        succ_per_channel = np.zeros(2)
        for i in range(2):
            norm_i = 2 * max(abs(vm_post[i]), abs(vm_pre[i]), epsilon)
            succ_per_channel[i] = (vm_post[i] - vm_pre[i]) / norm_i
    
        # Weighted average
        succ_star = np.sum(weights * succ_per_channel) / np.sum(weights)
    
        return float(np.clip(succ_star, -1.0, 1.0))
    
    def step(self, use_memory: bool) -> Tuple[bool, int, Dict]:
        """Execute one A1-A8 iteration."""
        
        # A1: Categorize
        category = self.get_category()
        
        # A2: Assess affect BEFORE action
        affect_pre = self.assess_affect_from_raw_state(self.y)
        z_need = affect_pre['z']
        h_need = self.compute_hints_from_needs()
        was_in_pain = affect_pre['in_pain']
        
        # A3: Memory retrieval
        if use_memory and len(self.memory) > 0:
            z_mem, h_mem, reliability = self.memory.retrieve(category)
        else:
            z_mem, h_mem, reliability = None, None, 0.0
        
        # A4: Fuse affect and hints, select policy
        z_fused = self.fuse_affect(z_need, z_mem, reliability)
        h_aff = self.compute_hints_from_affect(z_fused)
        h = self.fuse_hints(h_need, h_mem, h_aff, reliability)
        
        arousal = z_fused[2] if len(z_fused) > 2 else self.cfg['base_arousal']
        policy = self.select_policy(h, arousal, affect_pre['in_pain'])
        
        # A5: Policy instantiation
        desired_theta = self.get_action_for_policy(policy)
        
        # A6: Execute
        self.execute_action(desired_theta, arousal)
        
        # A7: Reappraise - assess affect AFTER action
        affect_post = self.assess_affect_from_raw_state(self.y)
        z_star = affect_post['z']
        now_in_pain = affect_post['in_pain']
        
        # Crossing = transition from no pain to pain (based on affect)
        crossed_into_harm = now_in_pain and not was_in_pain
        
        # Success from affect change
        succ_star = self.compute_success_from_affect(z_need, z_star)
        
        # A8: Store episode
        if use_memory:
            episode = Episode(
                c=category.copy(),
                z=z_need.copy(),
                h=h.copy(),
                z_star=z_star.copy(),
                succ_star=succ_star
            )
            self.memory.store(episode)
        
        debug_info = {
            'z_fused_harm': z_fused[0],
            'reliability': reliability,
            'n_episodes': len(self.memory),
        }
        
        return crossed_into_harm, policy, debug_info


# ============================================================================
# WORLD SIMULATION
# ============================================================================

class World:
    """Simulation world with harm zone at y > 0."""
    
    def __init__(self, cfg: Dict, use_memory: bool = True):
        self.cfg = cfg
        self.use_memory = use_memory
        self.N = cfg['num_agents']
        self._init_agents()
        self.history = {
            'positions': [],
            'crossings': [],
            'policies': [],
            'mean_episodes': [],
        }
        self.t = 0
    
    def _init_agents(self) -> None:
        """Initialize agents in safe zone."""
        self.agents = []
        bx = self.cfg['boundary_x']
        by = self.cfg['boundary_y']
        
        for _ in range(self.N):
            x = random.uniform(-bx * 0.7, bx * 0.7)
            y = random.uniform(-by * 0.8, -by * 0.4)
            theta = random.uniform(-math.pi, math.pi)
            self.agents.append(Agent(x, y, theta, self.cfg))
    
    def step(self) -> None:
        crossings = 0
        policies = []
        n_episodes = []
        
        for agent in self.agents:
            crossed, policy, debug = agent.step(self.use_memory)
            if crossed:
                crossings += 1
            policies.append(policy)
            n_episodes.append(debug['n_episodes'])
        
        self.history['positions'].append([(a.x, a.y) for a in self.agents])
        self.history['crossings'].append(crossings)
        self.history['policies'].append(policies)
        self.history['mean_episodes'].append(sum(n_episodes) / self.N)
        self.t += 1
    
    def run(self, steps: int, verbose: bool = True) -> None:
        for t in range(steps):
            self.step()
            
            if verbose and t % 100 == 0:
                total_crossings = sum(self.history['crossings'])
                recent_crossings = sum(self.history['crossings'][max(0, t - 50):t + 1])
                center_y = np.mean([a.y for a in self.agents])
                mean_mem = self.history['mean_episodes'][-1] if self.history['mean_episodes'] else 0
                
                print(f"Step {t:4d}: total={total_crossings:3d}, "
                      f"recent50={recent_crossings:2d}, "
                      f"center_y={center_y:+5.1f}, "
                      f"mean_mem={mean_mem:.1f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_animation(world: World, path: str, fps: int = 20, skip: int = 4) -> None:
    """Create animation of agent behavior."""
    print(f"Creating {path}...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bx, by = world.cfg['boundary_x'], world.cfg['boundary_y']
    n_bands = world.cfg['n_y_bands']
    
    ax.set_xlim(-bx * 1.05, bx * 1.05)
    ax.set_ylim(-by * 1.1, by * 1.1)
    ax.set_aspect('equal')
    
    # Harm zone
    ax.add_patch(Rectangle((-bx, 0), 2 * bx, by, alpha=0.2, color='red'))
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Band boundaries
    band_size = 2 * by / n_bands
    for i in range(1, n_bands):
        yl = -by + i * band_size
        ax.axhline(y=yl, color='blue', linestyle=':', alpha=0.3)
    
    # Highlight boundary band
    bb_bottom = -by + 2 * band_size
    bb_top = -by + 3 * band_size
    ax.add_patch(Rectangle((-bx, bb_bottom), 2 * bx, band_size,
                           alpha=0.15, color='orange'))
    ax.text(bx * 0.6, (bb_bottom + bb_top) / 2, 'BOUNDARY\nBAND',
            fontsize=10, color='orange', fontweight='bold', alpha=0.8,
            ha='center', va='center')
    
    ax.text(bx * 0.85, by * 0.6, 'HARM\n(shock)', fontsize=12, color='red',
            fontweight='bold', alpha=0.6, ha='center')
    ax.text(bx * 0.85, -by * 0.6, 'SAFE', fontsize=14, color='blue',
            fontweight='bold', alpha=0.6)
    
    mem_text = "WITH MEMORY" if world.use_memory else "NO MEMORY"
    ax.text(0, by * 0.9, mem_text, fontsize=16, ha='center',
            fontweight='bold', color='purple' if world.use_memory else 'gray')
    
    T = len(world.history['positions'])
    frames = list(range(0, T, skip))
    
    scatter = ax.scatter([], [], s=60, edgecolors='black', linewidth=0.5)
    title = ax.set_title('')
    
    def update(frame_idx):
        t = frames[frame_idx]
        positions = world.history['positions'][t]
        scatter.set_offsets(positions)
        
        colors = []
        for (x, y) in positions:
            if y > 0:
                colors.append('red')
            elif y > bb_bottom:
                colors.append('orange')
            else:
                colors.append('blue')
        scatter.set_color(colors)
        
        total = sum(world.history['crossings'][:t + 1])
        recent = sum(world.history['crossings'][max(0, t - 30):t + 1])
        title.set_text(f't={t}   total={total}   recent30={recent}')
        
        return scatter, title
    
    anim = FuncAnimation(fig, update, frames=len(frames), 
                         interval=1000 // fps, blit=False)
    anim.save(path, writer=FFMpegWriter(fps=fps))
    plt.close()
    print(f"Saved {path}")


def plot_comparison(w_mem: World, w_no: World, path: Optional[str] = None) -> None:
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cumulative crossings
    ax = axes[0, 0]
    cum_no = np.cumsum(w_no.history['crossings'])
    cum_mem = np.cumsum(w_mem.history['crossings'])
    ax.plot(cum_no, 'r-', label='No Memory', lw=2)
    ax.plot(cum_mem, 'b-', label='With Memory', lw=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Crossings')
    ax.set_title('Total Crossings into Harm Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Crossing rate
    ax = axes[0, 1]
    window = 40
    
    def rolling_sum(arr, w):
        return [sum(arr[max(0, i - w + 1):i + 1]) for i in range(len(arr))]
    
    rate_no = rolling_sum(w_no.history['crossings'], window)
    rate_mem = rolling_sum(w_mem.history['crossings'], window)
    ax.plot(rate_no, 'r-', label='No Memory', alpha=0.8)
    ax.plot(rate_mem, 'b-', label='With Memory', alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel(f'Crossings (last {window} steps)')
    ax.set_title('Crossing Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Memory size per agent
    ax = axes[1, 0]
    ax.plot(w_mem.history['mean_episodes'], 'b-', lw=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Episodes per Agent')
    ax.set_title('Memory Size per Agent\n(only significant episodes stored)')
    ax.grid(True, alpha=0.3)
    
    # Early vs Late
    ax = axes[1, 1]
    T = len(w_mem.history['crossings'])
    T_tenth = T // 10
    
    early_no = sum(w_no.history['crossings'][:T_tenth])
    late_no = sum(w_no.history['crossings'][-T_tenth:])
    early_mem = sum(w_mem.history['crossings'][:T_tenth])
    late_mem = sum(w_mem.history['crossings'][-T_tenth:])
    
    x_pos = [0, 1]
    bar_width = 0.35
    ax.bar([p - bar_width/2 for p in x_pos], [early_no, late_no], bar_width,
           label='No Memory', color='red', alpha=0.7)
    ax.bar([p + bar_width/2 for p in x_pos], [early_mem, late_mem], bar_width,
           label='With Memory', color='blue', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Early (0-10%)', 'Late (90-100%)'])
    ax.set_ylabel('Crossings')
    ax.set_title(f'Early vs Late\n'
                 f'No Mem: {early_no}→{late_no}   '
                 f'Mem: {early_mem}→{late_mem}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if path:
        plt.savefig(path, dpi=150)
        print(f"Saved {path}")
    plt.close()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(cfg: Dict, seeds: List[int], n_steps: int = 1500) -> Dict:
    """Run comparison experiment across multiple seeds."""
    results = {
        'no_mem': [], 'mem': [],
        'no_mem_early': [], 'mem_early': [],
        'no_mem_late': [], 'mem_late': [],
    }
    
    for i, seed in enumerate(seeds):
        print(f"--- Seed {seed} ---")
        
        random.seed(seed)
        np.random.seed(seed)
        w_no = World(cfg, use_memory=False)
        w_no.run(n_steps, verbose=False)
        
        random.seed(seed)
        np.random.seed(seed)
        w_mem = World(cfg, use_memory=True)
        w_mem.run(n_steps, verbose=False)
        
        T = len(w_mem.history['crossings'])
        T_tenth = T // 10
        
        c_no = sum(w_no.history['crossings'])
        c_mem = sum(w_mem.history['crossings'])
        early_no = sum(w_no.history['crossings'][:T_tenth])
        late_no = sum(w_no.history['crossings'][-T_tenth:])
        early_mem = sum(w_mem.history['crossings'][:T_tenth])
        late_mem = sum(w_mem.history['crossings'][-T_tenth:])
        
        results['no_mem'].append(c_no)
        results['mem'].append(c_mem)
        results['no_mem_early'].append(early_no)
        results['no_mem_late'].append(late_no)
        results['mem_early'].append(early_mem)
        results['mem_late'].append(late_mem)
        
        print(f"  No Memory: {c_no:3d} total (early={early_no:2d}, late={late_no:2d})")
        print(f"  With Mem:  {c_mem:3d} total (early={early_mem:2d}, late={late_mem:2d})")
        
        if i == 0:
            print("\n  Generating visualizations...")
            import os
            os.makedirs('outputs', exist_ok=True)
            plot_comparison(w_mem, w_no, 'outputs/harm-algorithm_comparison.png')
            create_animation(w_no, 'outputs/harm-no_memory.mp4', skip=4)
            create_animation(w_mem, 'outputs/harm-with_memory.mp4', skip=4)
            print()
    
    return results


def print_summary(results: Dict) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTOTAL CROSSINGS:")
    print(f"  No Memory:   {np.mean(results['no_mem']):5.1f} ± {np.std(results['no_mem']):4.1f}")
    print(f"  With Memory: {np.mean(results['mem']):5.1f} ± {np.std(results['mem']):4.1f}")
    
    print(f"\nEARLY (first 10%):")
    print(f"  No Memory:   {np.mean(results['no_mem_early']):5.1f} ± {np.std(results['no_mem_early']):4.1f}")
    print(f"  With Memory: {np.mean(results['mem_early']):5.1f} ± {np.std(results['mem_early']):4.1f}")
    
    print(f"\nLATE (last 10%):")
    print(f"  No Memory:   {np.mean(results['no_mem_late']):5.1f} ± {np.std(results['no_mem_late']):4.1f}")
    print(f"  With Memory: {np.mean(results['mem_late']):5.1f} ± {np.std(results['mem_late']):4.1f}")
    
    if np.mean(results['no_mem_late']) > 0:
        reduction = ((np.mean(results['no_mem_late']) - np.mean(results['mem_late'])) 
                     / np.mean(results['no_mem_late']) * 100)
        print(f"\n  Late crossing reduction: {reduction:.0f}%")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AFFECT-BASED EPISODIC MEMORY DEMO")
    print("Algorithm A1-A8: Anticipatory Harm Avoidance")
    print("=" * 70)
    
    cfg = DEFAULT_CONFIG.copy()
    band_size = 2 * cfg['boundary_y'] / cfg['n_y_bands']
    
    print(f"\nConfiguration:")
    print(f"  Agents: {cfg['num_agents']}")
    print(f"  World: x ∈ [-{cfg['boundary_x']}, +{cfg['boundary_x']}], "
          f"y ∈ [-{cfg['boundary_y']}, +{cfg['boundary_y']}]")
    print(f"  Harm zone: y > 0 (electric shock)")
    print(f"  Pain threshold: harm_valence < {cfg['pain_threshold']}")
    print(f"  Y-bands: {cfg['n_y_bands']} (band size: {band_size})")
    print(f"  Boundary band: y ∈ [{-cfg['boundary_y'] + 2*band_size:.1f}, "
          f"{-cfg['boundary_y'] + 3*band_size:.1f}]")
    print(f"  Store threshold: |succ*| > {cfg['store_threshold']}")
    print(f"  Emotional drives: curiosity={cfg['curiosity_intensity']}, "
          f"harm={cfg['harm_intensity']}")
    print()

    import sys
    sys.stdout.flush()
    
    seeds = [ 456, 516, 637, 789, 101]
    results = run_experiment(cfg, seeds, n_steps=4000)
    print_summary(results)
    
    print("\nDone!")
