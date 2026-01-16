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

Full implementation of Algorithm A1-A8 with consistent normalization,
bounded hint fusion, and configurable success measures.

Scenario:
    Agents explore a 2D world with a harm zone (y > 0) that delivers
    negative affect ("electric shock") upon contact. No perceptual warning
    signal exists—agents must learn from experience.

Key Demonstration:
    - Without memory: Agents wander freely, repeatedly entering the harm
      zone and experiencing reactive negative affect.
    - With memory: Agents retrieve affect from similar past situations,
      triggering anticipatory avoidance BEFORE crossing into harm.

Key Mechanism:
    Memory retrieval uses coarse categorical keys (y-bands, velocity sign,
    harm flag) while affect arises from raw sensation. An agent approaching
    y = 0 from below retrieves memories from y > 0 (same band), receiving
    anticipatory warning that immediate perception cannot provide.

Normalization Strategy :
    - H_need and H_affect matrices are row-L1 normalized so that hints
      h = H @ x remain bounded in [-1, 1] when inputs are in [-1, 1].
    - Hint fusion weights (alpha_need, alpha_mem, alpha_aff) are renormalized
      to sum to 1 at each step, ensuring fused hints stay in [-1, 1].
    - Memory retrieval uses h_mem = Σ w_j * succ_j * h_j (no denominator
      normalization), so success magnitude scales hint strength linearly.

Credit Assignment:
    One-hot policy tagging isolates credit to the responsible policy.
    At action selection, the "responsible policy" is identified as
    π_hat = argmax_π [ q(π) * s̃_π(u_t) ], and only that policy's one-hot
    tag is stored. This prevents spurious credit leakage across policies.

Success Measures (SI equations):
    - Drive reduction: succ_drive = (||d||₁ - ||d*||₁) / max(||d||₁, ||d*||₁, ε)
    - Weighted emotion: succ_emotion = Σᵢ wᵢ * (v*ᵢm*ᵢ - vᵢmᵢ) / (2 * max(...))
    - Hybrid: succ = ω * succ_drive + (1-ω) * succ_emotion
    Success is computed from NEED-BASED affect (z^need, z*^need), not fused
    affect, preventing "fear relief" artifacts when memory injects anticipatory
    valence into z_t.

Category Design:
    c_t = [y_band one-hot, sign(v_y) one-hot, harm_flag one-hot]
    This expanded representation reduces spurious cosine-similarity ties
    that would otherwise make K=1 retrieval unreliable.

Physics Model:
    State: (pos_x, pos_y, vel_x, vel_y) in bounded 2D arena
    Actions: discrete acceleration directions plus coast
    Dynamics: acceleration noise, velocity drag, boundary reflection

Needs (2 channels):
    - Safety: 1.0 in safe zone (y ≤ 0), 0.0 in harm zone (y > 0)
    - Motion: speed / target_speed, clamped to [0, 1]

Policies (2):
    - Flee: accelerate south (innate safe direction)
    - Thrust: accelerate along current velocity (maintain motion)

Algorithm Mapping (A1-A8):
    A1: x_t ← observe(); (c_t, y_t) ← categorize(x_t)
    A2: n_t ← assess_needs(); z^need_t ← affect(n_t); h^need_t ← H_need @ d_t
    A3: (z^mem_t, h^mem_t, r_t) ← memory.retrieve(c_t)
    A4: z_t ← fuse(z^need_t, z^mem_t; r_t)
        h^aff_t ← H_affect @ z_t
        h_t ← fuse(h^need_t, h^mem_t, h^aff_t; r_t)
        q_t(π) ← softmax(h_t / τ₁(a_t))
    A5: s_t(u) ← Σ_π q_t(π) * s̃_π(u); u_t ← select(s_t, τ₂(a_t))
    A6: execute(u_t)
    A7: (z*_t, succ*_t) ← reappraise(z^need_t, z*^need_t)
    A8: store episode if |succ*_t| ≥ threshold

Design Notes:
    - Policy-to-action mappings (Flee → south) are innate. The system learns
      WHEN to activate policies, not HOW to execute them.
    - Only significant episodes (|succ*| ≥ threshold) are stored to prevent
      memory dilution from neutral wandering.
    - Reliability r_t (max retrieval similarity) modulates memory influence:
      low reliability → trust needs more; high reliability → trust memory.
    - Temperature schedules τ₁(a), τ₂(a) decrease with arousal, making
      high-arousal states more decisive and low-arousal states more exploratory.

Expected Results:
    - Early phase: Similar crossing rates (learning period)
    - Late phase: ~0 crossings with memory vs. continued crossings without
    - Reduction: 80-100% fewer total crossings with memory enabled

Usage:
    python affect_memory_demo.py

Outputs:
    - outputs/comparison.png: Side-by-side metrics
    - outputs/no_memory.mp4: Animation without episodic memory
    - outputs/with_memory.mp4: Animation with episodic memory

Reference:
    "Synthetic Emotions and Consciousness: Exploring Architectural Boundaries"
    See paper Figure 1 and Supplementary Information (SI) Part I.

Repository:
    https://github.com/affect-based-control/synthetic-emotion-controller
"""
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.animation import FuncAnimation, FFMpegWriter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import deque
import warnings
warnings.filterwarnings("ignore")

EPS = 1e-12


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def clip11(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, -1.0, 1.0)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable softmax with temperature.
      - lower temperature => more peaked (more deterministic)
      - higher temperature => flatter (more exploratory)
    """
    values = np.asarray(values, dtype=float)
    temperature = max(float(temperature), 1e-6)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted / temperature)
    return exp_values / (exp_values.sum() + EPS)

def l2_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=float).flatten()
    n = float(np.linalg.norm(x))
    if n < eps:
        return np.zeros_like(x)
    return x / (n + eps)


def normalize_rows_l1(M: np.ndarray) -> np.ndarray:
    """
    Row-wise L1 normalization. If x components are in [-1,1], then each row output is in [-1,1].
    """
    M = np.asarray(M, dtype=float)
    row_norm = np.sum(np.abs(M), axis=1, keepdims=True) + EPS
    return M / row_norm


def renorm_weights(*alphas: float) -> Tuple[float, ...]:
    s = float(sum(alphas)) + EPS
    return tuple(float(a) / s for a in alphas)


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n, dtype=float)
    v[int(i)] = 1.0
    return v


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    na = float(np.linalg.norm(a)) + EPS
    nb = float(np.linalg.norm(b)) + EPS
    return float(np.dot(a, b) / (na * nb))


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG: Dict = {
    # Population / world
    "num_agents": 30,
    "world_bound_x": 20.0,
    "world_bound_y": 10.0,
    "num_y_bands": 6,  # even helps reduce "straddle y=0" ambiguity
    "time_step": 0.2,

    # Motion dynamics
    "max_speed": 3.0,
    "target_speed": 1.5,
    "accel_magnitude": 2.0,
    "accel_noise_std": 0.4,
    "velocity_drag": 0.03,

    # Action space
    "num_accel_directions": 8,

    # Policies
    "policy_names": ["Flee", "Thrust"],
    "num_policies": 2,
    "POLICY_FLEE": 0,
    "POLICY_THRUST": 1,

    # Need targets n^◇
    "need_target_safety": 0.9,
    "need_target_motion": 0.6,

    # Drive gain α (per-need)
    "drive_gain_safety": 4.0,
    "drive_gain_motion": 2.5,

    # Valence scaling λ (SI: v = -λ ⊙ d). Keep <=1 to preserve bounds.
    "valence_lambda": [1.0, 1.0],

    # Arousal model
    "arousal_base": 0.3,
    "arousal_scale": 0.5,  # arousal = base + scale * mean(|d|)

    # H_need: [num_policies x num_needs] applied to drives d (bounded in [-1,1])
    # NOTE: Will be row-L1 normalized at runtime.
    "H_need": [
        [ 2.0, -0.2],   # Flee vs [drive_safety, drive_motion]
        [-0.3,  1.5],   # Thrust
    ],

    # H_affect: [num_policies x d_aff] applied to affect input vector.
    # Supported input conventions by column count:
    #   3 columns: [v_safety, v_motion, arousal]
    #   5 columns: [v_safety, v_motion, arousal, d_safety, d_motion]
    #   7 columns: [v_safety, v_motion, m_safety, m_motion, arousal, d_safety, d_motion]
    # NOTE: Will be row-L1 normalized at runtime.
    "H_affect": [
        [-2.0, -0.1,  0.3],
        [ 0.2,  1.0, -0.1],
    ],

    # Hint fusion weights (renormalized each step to sum to 1)
    "alpha_hint_need": 0.30,
    "alpha_hint_memory": 0.60,
    "alpha_hint_affect": 0.40,

    # Affect fusion z_t = Fuse(z_need, z_mem; reliability)
    "beta_affect_fusion": 0.40,  # larger => trust needs more by default

    # Policy temperature tau_1(a): decreases with arousal => more decisive
    "tau_policy_base": 0.40,
    "tau_policy_arousal_scale": 0.25,
    "tau_policy_min": 0.05,

    # Action temperature tau_2(a): decreases with arousal => more decisive
    "tau_action_base": 0.35,
    "tau_action_arousal_scale": 0.20,
    "tau_action_min": 0.05,

    # Policy templates
    "flee_direction": -math.pi / 2,  # south
    "policy_template_sharpness": 3.0,

    # Episodic memory
    "memory_capacity": 100,
    "memory_retrieval_k": 1,              # user wants K=1
    "memory_retrieval_temperature": 0.25, # softmax temperature over similarities (lower => more peaked)
    "memory_tag_alpha": 0.15,             # affect tag blend between (z_t) and (z*_t)
    "memory_store_threshold": 0.25,       # store if |succ*| >= threshold

    # One-hot tagging option (SI optional)
    "memory_store_one_hot_policy_tag": True,

    # Category design (reduces similarity ties)
    "category_include_vy_sign": True,
    "category_include_harm_flag": True,

    # Success weights w_i for hedonic aggregation
    "success_weight_safety": 4.0,
    "success_weight_motion": 1.0,

    # Success mode: "drive", "emotion", or "hybrid"
    "success_mode": "hybrid",
    "success_omega": 0.50,  # omega in hybrid success

    # Action selection
    "action_use_argmax_probability": 0.80,
}


# ============================================================================
# AFFECT COMPUTATION (SI-consistent components)
# ============================================================================

def affect_from_needs(
    needs: np.ndarray,
    cfg: Dict
) -> Dict[str, np.ndarray]:
    """
    Given need levels n in [0,1]^2:
      d = tanh(alpha * (n^◇ - n))      drives in [-1,1]^2
      v = -lambda ⊙ d                 valence in [-1,1]^2 (after clip)
      m = |d|                          magnitude in [0,1]^2
      a = clip(base + scale * mean(m), 0, 1) arousal in [0,1]
    Returns a canonical z vector of length 7:
      z = [v_safety, v_motion, m_safety, m_motion, a, d_safety, d_motion]
    """
    needs = np.asarray(needs, dtype=float).flatten()
    if needs.shape[0] != 2:
        raise ValueError(f"needs must be length 2, got {needs.shape}")

    n_safety, n_motion = float(needs[0]), float(needs[1])
    n_target = np.array([cfg["need_target_safety"], cfg["need_target_motion"]], dtype=float)

    alpha = np.array([cfg["drive_gain_safety"], cfg["drive_gain_motion"]], dtype=float)
    d = np.tanh(alpha * (n_target - np.array([n_safety, n_motion], dtype=float)))  # [-1,1]^2

    lam = np.array(cfg["valence_lambda"], dtype=float).flatten()
    if lam.shape[0] != 2:
        raise ValueError("valence_lambda must have length 2")
    # keep valence in [-1,1] (lambda should ideally be <= 1, but clip just in case)
    v = clip11(-lam * d)

    m = np.abs(d)  # [0,1]^2
    a = clip(cfg["arousal_base"] + cfg["arousal_scale"] * float(np.mean(m)), 0.0, 1.0)

    z = np.array([v[0], v[1], m[0], m[1], a, d[0], d[1]], dtype=float)
    # z components are each within [-1,1] by construction/clipping
    z = clip11(z)

    return {
        "n": np.array([n_safety, n_motion], dtype=float),
        "d": d,
        "v": v,
        "m": m,
        "a": np.array([a], dtype=float),
        "z": z,
    }


# ============================================================================
# SUCCESS MEASURES (as in user's SI excerpt)
# ============================================================================

def success_drive_reduction(d_pre: np.ndarray, d_post: np.ndarray, eps: float = 1e-6) -> float:
    """
    succ_drive = (||d||_1 - ||d*||_1) / max(||d||_1, ||d*||_1, eps) in [-1,1]
    """
    d_pre = np.asarray(d_pre, dtype=float).flatten()
    d_post = np.asarray(d_post, dtype=float).flatten()
    n1_pre = float(np.sum(np.abs(d_pre)))
    n1_post = float(np.sum(np.abs(d_post)))
    denom = max(n1_pre, n1_post, eps)
    return clip((n1_pre - n1_post) / denom, -1.0, 1.0)


def success_weighted_emotion(v_pre: np.ndarray, m_pre: np.ndarray,
                            v_post: np.ndarray, m_post: np.ndarray,
                            w: np.ndarray, eps: float = 1e-6) -> float:
    """
    Per-channel:
      succ_i = (v*_i m*_i - v_i m_i) / (2 * max(|v*_i m*_i|, |v_i m_i|, eps)) in [-1,1]
    Aggregate:
      succ_emotion = sum_i w_i succ_i / sum_i w_i in [-1,1]
    """
    v_pre = np.asarray(v_pre, dtype=float).flatten()
    m_pre = np.asarray(m_pre, dtype=float).flatten()
    v_post = np.asarray(v_post, dtype=float).flatten()
    m_post = np.asarray(m_post, dtype=float).flatten()
    w = np.asarray(w, dtype=float).flatten()

    if not (v_pre.shape == m_pre.shape == v_post.shape == m_post.shape == w.shape):
        raise ValueError("v/m shapes and weights must match for success_weighted_emotion")

    succ_i = np.zeros_like(v_pre, dtype=float)
    for i in range(len(v_pre)):
        pre = float(v_pre[i] * m_pre[i])
        post = float(v_post[i] * m_post[i])
        denom = 2.0 * max(abs(post), abs(pre), eps)
        succ_i[i] = (post - pre) / denom

    denom_w = float(np.sum(w)) + EPS
    succ = float(np.sum(w * succ_i) / denom_w)
    return clip(succ, -1.0, 1.0)


def success_hybrid(succ_drive: float, succ_emotion: float, omega: float) -> float:
    """
    succ* = omega * succ_drive + (1-omega) * succ_emotion, omega in [0,1]
    """
    omega = clip(float(omega), 0.0, 1.0)
    return clip(omega * float(succ_drive) + (1.0 - omega) * float(succ_emotion), -1.0, 1.0)


# ============================================================================
# EPISODE DATA STRUCTURE (A8)
# ============================================================================

@dataclass
class Episode:
    category: np.ndarray      # c_t
    affect_fused: np.ndarray  # z_t
    hints_fused: np.ndarray   # h_t (stored for completeness)
    policy_tag: np.ndarray    # one-hot responsible-policy tag (optional in SI)
    affect_post: np.ndarray   # z*_t (from post-act needs)
    success_post: float       # succ*_t in [-1,1]


# ============================================================================
# EPISODIC MEMORY SYSTEM (A3, A8)
# ============================================================================

class EpisodicMemory:
    def __init__(self, cfg: Dict):
        self.capacity = int(cfg["memory_capacity"])
        self.retrieval_k = int(cfg["memory_retrieval_k"])
        self.retrieval_tau = float(cfg["memory_retrieval_temperature"])
        self.tag_alpha = float(cfg["memory_tag_alpha"])
        self.store_threshold = float(cfg["memory_store_threshold"])
        self.num_policies = int(cfg["num_policies"])
        self.tau_ret = float(cfg.get("memory_retrieval_tau_ret", 1.0))
        self.min_similarity = float(cfg.get("memory_min_similarity", 0.0))
        self.episodes: deque = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.episodes)


    def store(self, episode: Episode) -> bool:
        """A8: store if |succ*| >= threshold; also L2-normalize key for SI dot-product similarity."""
        if abs(float(episode.success_post)) < self.store_threshold:
            return False

        # Normalize category key once, so sim(c_t, c_j) = c_t^T c_j matches SI
        episode.category = l2_normalize(episode.category)
    
        self.episodes.append(episode)
        return True

    def retrieve(self, category_current: np.ndarray):
        """
        A3: retrieval:
          sim = c_t^T c_j with L2-normalized keys
          w = softmax( tau_ret * sim )
          h_mem = Σ w_j succ*_j h_j
          z_mem = Σ w_j z_tag_j, z_tag_j = α z_j + (1-α) z*_j
          abstain if no suitable episodes (e.g., max sim < min_similarity)
        """
        if len(self.episodes) == 0:
            return None, None, 0.0

        c_t = l2_normalize(category_current)
    
        # dot-product similarity because all stored keys are L2-normalized
        sims = np.array([float(np.dot(c_t, ep.category)) for ep in self.episodes], dtype=float)
    
        k = min(self.retrieval_k, len(self.episodes))
        top_idx = np.argsort(sims)[-k:]
        top_sims = sims[top_idx]
    
        reliability = float(np.max(top_sims)) if top_sims.size else 0.0
        reliability = float(np.clip(reliability, 0.0, 1.0))
    
        # Optional abstention if no sufficiently similar neighbor exists
        if reliability < getattr(self, "min_similarity", 0.0):
            # SI: memory abstains -> return neutral aggregates
            z_dim = int(self.episodes[0].affect_fused.shape[0])
            return np.zeros(z_dim, dtype=float), np.zeros(self.num_policies, dtype=float), 0.0

        # SI weights: w = softmax( τ_ret * sim )
        tau_ret = float(getattr(self, "tau_ret", 1.0))
        w = softmax(tau_ret * top_sims, temperature=1.0)  # sums to 1

        # z_mem = Σ w_j z_tag_j (no success factor)
        z_dim = int(self.episodes[0].affect_fused.shape[0])
        z_mem = np.zeros(z_dim, dtype=float)
        for idx, wi in zip(top_idx, w):
            ep = self.episodes[int(idx)]
            z_tag = self.tag_alpha * ep.affect_fused + (1.0 - self.tag_alpha) * ep.affect_post
            z_mem += float(wi) * z_tag
        z_mem = np.clip(z_mem, -1.0, 1.0)

        # h_mem = Σ w_j succ*_j h_j  (SI eq. hmem-norm)
        h_mem = np.zeros(self.num_policies, dtype=float)
        for idx, wi in zip(top_idx, w):
            ep = self.episodes[int(idx)]
            h_mem += float(wi) * float(ep.success_post) * ep.policy_tag
        h_mem = np.clip(h_mem, -1.0, 1.0)

        return z_mem, h_mem, reliability


# ============================================================================
# AGENT (A1-A8)
# ============================================================================

class Agent:
    def __init__(self, pos_x: float, pos_y: float, vel_x: float, vel_y: float, cfg: Dict):
        self.pos_x = float(pos_x)
        self.pos_y = float(pos_y)
        self.vel_x = float(vel_x)
        self.vel_y = float(vel_y)
        self.cfg = cfg

        self.memory = EpisodicMemory(cfg)
        self.actions = self._build_action_space()
        self.num_actions = len(self.actions)

        # Normalize H matrices so hints are bounded in [-1,1]
        self.H_need = normalize_rows_l1(np.array(cfg["H_need"], dtype=float))
        self.H_affect = normalize_rows_l1(np.array(cfg["H_affect"], dtype=float))

        # Sanity check dimensions
        if self.H_need.shape != (cfg["num_policies"], 2):
            raise ValueError(f"H_need shape must be ({cfg['num_policies']}, 2), got {self.H_need.shape}")

    def _build_action_space(self) -> List[Tuple[float, float]]:
        actions: List[Tuple[float, float]] = []
        magnitude = float(self.cfg["accel_magnitude"])
        num_directions = int(self.cfg["num_accel_directions"])
        for i in range(num_directions):
            angle = 2.0 * math.pi * i / num_directions
            actions.append((magnitude * math.cos(angle), magnitude * math.sin(angle)))
        actions.append((0.0, 0.0))  # Coast
        return actions

    # ---------------------------
    # A1: Observe + Categorize
    # ---------------------------

    def observe(self) -> Dict:
        """A1 line 1: x_t <- observe(W_t)"""
        speed = math.sqrt(self.vel_x**2 + self.vel_y**2)
        return {
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "vel_x": self.vel_x,
            "vel_y": self.vel_y,
            "speed": speed,
            "in_harm_zone": self.pos_y > 0.0,
        }

    def categorize(self, obs: Dict) -> Tuple[np.ndarray, Dict]:
        """A1 line 2: (c_t, y_t) <- categorize(x_t)"""

        # y-band one-hot
        num_bands = int(self.cfg["num_y_bands"])
        bound_y = float(self.cfg["world_bound_y"])
        band_height = 2.0 * bound_y / num_bands
        band_index = int((obs["pos_y"] + bound_y) / band_height)
        band_index = int(clip(band_index, 0, num_bands - 1))

        c_band = np.zeros(num_bands, dtype=float)
        c_band[band_index] = 1.0

        # optional: sign(v_y) one-hot
        extra = []
        if bool(self.cfg.get("category_include_vy_sign", True)):
            vy_up = 1 if obs["vel_y"] > 0.0 else 0
            c_vy = np.zeros(2, dtype=float)
            c_vy[vy_up] = 1.0
            extra.append(c_vy)

        # optional: harm flag one-hot
        if bool(self.cfg.get("category_include_harm_flag", True)):
            harm = 1 if obs["pos_y"] > 0.0 else 0
            c_h = np.zeros(2, dtype=float)
            c_h[harm] = 1.0
            extra.append(c_h)

        category = np.concatenate([c_band] + extra) if extra else c_band

        # y_t / situation parameters for policy templates
        y_t = {
            "vel_x": float(obs["vel_x"]),
            "vel_y": float(obs["vel_y"]),
            "speed": float(obs["speed"]),
        }
        return category, y_t

    # ---------------------------
    # A2: Need appraisal -> affect, need-hints
    # ---------------------------

    def assess_needs(self, obs: Dict) -> np.ndarray:
        """A2 line 3: n_t <- assess_needs(...)"""
        need_safety = 0.0 if bool(obs["in_harm_zone"]) else 1.0
        need_motion = clip(float(obs["speed"]) / float(self.cfg["target_speed"]), 0.0, 1.0)
        return np.array([need_safety, need_motion], dtype=float)

    def hints_from_needs(self, d_t: np.ndarray) -> np.ndarray:
        """A2 line 5: h^need_t <- H_need d_t (drives)"""
        h = self.H_need @ np.asarray(d_t, dtype=float).flatten()
        return clip11(h)

    # ---------------------------
    # A4: Affect and hint fusion -> policy probabilities
    # ---------------------------

    def fuse_affect(self, z_need: np.ndarray, z_mem: Optional[np.ndarray], reliability: float) -> np.ndarray:
        """A4 line 9: z_t <- Fuse(z^need_t, z^mem_t)"""
        if z_mem is None:
            return clip11(np.asarray(z_need, dtype=float).copy())

        z_need = np.asarray(z_need, dtype=float).flatten()
        z_mem = np.asarray(z_mem, dtype=float).flatten()
        if z_need.shape != z_mem.shape:
            raise ValueError(f"z_need and z_mem must match shapes, got {z_need.shape} vs {z_mem.shape}")

        r = clip(float(reliability), 0.0, 1.0)
        beta = float(self.cfg["beta_affect_fusion"])
        # As reliability increases, trust memory more
        w_need = beta + (1.0 - beta) * (1.0 - r)
        w_mem = (1.0 - beta) * r
        w_need, w_mem = renorm_weights(w_need, w_mem)

        z = w_need * z_need + w_mem * z_mem
        return clip11(z)

    def _affect_input_for_H_affect(self, z_fused: np.ndarray) -> np.ndarray:
        """
        Build affect input vector matching H_affect column count.
        Canonical z (len 7):
          [v0, v1, m0, m1, a, d0, d1]
        """
        z = np.asarray(z_fused, dtype=float).flatten()
        if z.shape[0] != 7:
            raise ValueError(f"Expected fused z length 7, got {z.shape[0]}")
        cols = int(self.H_affect.shape[1])

        if cols == 3:
            # [v_safety, v_motion, arousal]
            return np.array([z[0], z[1], z[4]], dtype=float)
        if cols == 5:
            # [v_safety, v_motion, arousal, d_safety, d_motion]
            return np.array([z[0], z[1], z[4], z[5], z[6]], dtype=float)
        if cols == 7:
            return z.copy()

        raise ValueError(f"Unsupported H_affect column count: {cols} (supported: 3,5,7)")

    def hints_from_affect(self, z_fused: np.ndarray) -> np.ndarray:
        """A4 line 10: h^aff_t <- H_affect z_t (or projection thereof)"""
        x = self._affect_input_for_H_affect(z_fused)
        h = self.H_affect @ x
        return clip11(h)

    def fuse_hints(self,
                   h_need: np.ndarray,
                   h_mem: Optional[np.ndarray],
                   h_aff: np.ndarray,
                   reliability: float) -> np.ndarray:
        """A4 line 11: h_t <- fuse(h^need_t, h^mem_t, h^aff_t) with renormalized weights"""
        h_need = clip11(np.asarray(h_need, dtype=float).flatten())
        h_aff = clip11(np.asarray(h_aff, dtype=float).flatten())
        h_mem_vec = clip11(np.asarray(h_mem, dtype=float).flatten()) if h_mem is not None else None

        alpha_need = float(self.cfg["alpha_hint_need"])
        alpha_aff = float(self.cfg["alpha_hint_affect"])
        alpha_mem = float(self.cfg["alpha_hint_memory"]) * clip(float(reliability), 0.0, 1.0) if h_mem_vec is not None else 0.0

        alpha_need, alpha_aff, alpha_mem = renorm_weights(alpha_need, alpha_aff, alpha_mem)

        h = alpha_need * h_need + alpha_aff * h_aff
        if h_mem_vec is not None:
            h += alpha_mem * h_mem_vec

        return clip11(h)

    def policy_probabilities(self, h_fused: np.ndarray, arousal: float) -> np.ndarray:
        """A4 line 12: q_t(pi) <- softmax(h_t / tau_1(a_t))"""
        a = clip(float(arousal), 0.0, 1.0)
        tau = max(float(self.cfg["tau_policy_base"]) - float(self.cfg["tau_policy_arousal_scale"]) * a,
                  float(self.cfg["tau_policy_min"]))
        return softmax(np.asarray(h_fused, dtype=float), temperature=tau)

    # ---------------------------
    # A5: Policy instantiation -> action scores; select action
    # ---------------------------

    def score_actions_for_policy(self, policy_index: int, y_t: Dict) -> np.ndarray:
        """A5 line 13: s~_pi(u)"""
        scores = np.zeros(self.num_actions, dtype=float)
        kappa = float(self.cfg["policy_template_sharpness"])

        if policy_index == int(self.cfg["POLICY_FLEE"]):
            flee_dir = float(self.cfg["flee_direction"])
            fx, fy = math.cos(flee_dir), math.sin(flee_dir)

            for i, (ax, ay) in enumerate(self.actions):
                an = math.sqrt(ax**2 + ay**2)
                if an < 1e-6:
                    scores[i] = 0.30
                else:
                    alignment = (ax * fx + ay * fy) / an  # in [-1,1]
                    scores[i] = math.exp(kappa * (alignment - 1.0))  # in (0,1]

        elif policy_index == int(self.cfg["POLICY_THRUST"]):
            vx, vy = float(y_t["vel_x"]), float(y_t["vel_y"])
            speed = float(y_t["speed"])

            if speed < 0.1:
                for i, (ax, ay) in enumerate(self.actions):
                    scores[i] = 0.50 if math.sqrt(ax**2 + ay**2) > 0.1 else 0.30
            else:
                ux, uy = vx / speed, vy / speed
                for i, (ax, ay) in enumerate(self.actions):
                    an = math.sqrt(ax**2 + ay**2)
                    if an < 1e-6:
                        scores[i] = 0.50
                    else:
                        alignment = (ax * ux + ay * uy) / an  # [-1,1]
                        scores[i] = math.exp(kappa * (alignment - 1.0) / 2.0)  # (0,1]

        return clip01(scores)

    def compute_action_scores(self, q_pi: np.ndarray, y_t: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """A5 line 14: s_t(u) = sum_pi q(pi) s~_pi(u). Also returns per-policy scores for credit assignment."""
        q = np.asarray(q_pi, dtype=float).flatten()
        if q.shape[0] != int(self.cfg["num_policies"]):
            raise ValueError("policy prob vector has wrong length")

        policy_scores_mat = np.zeros((int(self.cfg["num_policies"]), self.num_actions), dtype=float)
        s = np.zeros(self.num_actions, dtype=float)

        for pi in range(int(self.cfg["num_policies"])):
            ps = self.score_actions_for_policy(pi, y_t)
            policy_scores_mat[pi, :] = ps
            s += float(q[pi]) * ps

        return clip01(s), clip01(policy_scores_mat)

    def select_action(self, s_u: np.ndarray, arousal: float) -> int:
        """A5 line 15: select u_t via softmax(s_t / tau_2(a_t)) or argmax."""
        a = clip(float(arousal), 0.0, 1.0)
        tau = max(float(self.cfg["tau_action_base"]) - float(self.cfg["tau_action_arousal_scale"]) * a,
                  float(self.cfg["tau_action_min"]))
        probs = softmax(np.asarray(s_u, dtype=float), temperature=tau)

        if random.random() < float(self.cfg["action_use_argmax_probability"]):
            return int(np.argmax(s_u))
        return int(random.choices(range(self.num_actions), weights=probs)[0])

    # ---------------------------
    # A6: Execute
    # ---------------------------

    def execute_action(self, action_index: int) -> None:
        """A6 line 16: execute u_t"""
        ax, ay = self.actions[int(action_index)]
        dt = float(self.cfg["time_step"])

        # acceleration noise
        ax += random.gauss(0.0, float(self.cfg["accel_noise_std"]))
        ay += random.gauss(0.0, float(self.cfg["accel_noise_std"]))

        # update velocity
        self.vel_x += ax * dt
        self.vel_y += ay * dt

        # drag
        drag = float(self.cfg["velocity_drag"])
        self.vel_x *= (1.0 - drag)
        self.vel_y *= (1.0 - drag)

        # speed cap
        speed = math.sqrt(self.vel_x**2 + self.vel_y**2)
        max_speed = float(self.cfg["max_speed"])
        if speed > max_speed:
            scale = max_speed / (speed + EPS)
            self.vel_x *= scale
            self.vel_y *= scale

        # update position
        self.pos_x += self.vel_x * dt
        self.pos_y += self.vel_y * dt

        # reflect at boundaries
        bx, by = float(self.cfg["world_bound_x"]), float(self.cfg["world_bound_y"])
        if self.pos_x > bx - 0.1:
            self.pos_x = bx - 0.1
            self.vel_x = -abs(self.vel_x) * 0.8
        elif self.pos_x < -bx + 0.1:
            self.pos_x = -bx + 0.1
            self.vel_x = abs(self.vel_x) * 0.8

        if self.pos_y > by - 0.1:
            self.pos_y = by - 0.1
            self.vel_y = -abs(self.vel_y) * 0.8
        elif self.pos_y < -by + 0.1:
            self.pos_y = -by + 0.1
            self.vel_y = abs(self.vel_y) * 0.8

    # ---------------------------
    # A7: Reappraise + Success
    # ---------------------------

    def compute_success(self, affect_pre_need: Dict, affect_post_need: Dict) -> Tuple[np.ndarray, float]:
        """
        A7 line 18: (z*_t, succ*_t) <- reappraise(...)
        Here z*_t is taken as post-act need-based affect (SI excerpt).
        succ*_t computed by selected success_mode.
        """
        d_pre = affect_pre_need["d"]
        d_post = affect_post_need["d"]

        v_pre = affect_pre_need["v"]
        v_post = affect_post_need["v"]

        m_pre = affect_pre_need["m"]
        m_post = affect_post_need["m"]

        w = np.array([self.cfg["success_weight_safety"], self.cfg["success_weight_motion"]], dtype=float)

        succ_drive = success_drive_reduction(d_pre, d_post, eps=1e-6)
        succ_emotion = success_weighted_emotion(v_pre, m_pre, v_post, m_post, w=w, eps=1e-6)

        mode = str(self.cfg.get("success_mode", "hybrid")).lower()
        if mode == "drive":
            succ = succ_drive
        elif mode == "emotion":
            succ = succ_emotion
        elif mode == "hybrid":
            succ = success_hybrid(succ_drive, succ_emotion, omega=float(self.cfg.get("success_omega", 0.5)))
        else:
            raise ValueError(f"Unknown success_mode: {self.cfg.get('success_mode')}")

        return affect_post_need["z"], clip(float(succ), -1.0, 1.0)

    # ---------------------------
    # Full A1-A8 step
    # ---------------------------

    def step(self, use_memory: bool = True) -> Tuple[bool, int, Dict]:
        """
        Executes one complete A1-A8 iteration.
        Returns:
          crossed_into_harm (bool),
          policy_argmax (int),
          debug dict
        """
        # =========
        # A1
        # =========
        obs = self.observe()
        c_t, y_t = self.categorize(obs)
        was_in_harm = bool(obs["in_harm_zone"])

        # =========
        # A2
        # =========
        n_t = self.assess_needs(obs)
        affect_need = affect_from_needs(n_t, self.cfg)     # z^need_t, includes drives d_t
        z_need = affect_need["z"]
        h_need = self.hints_from_needs(affect_need["d"])

        # =========
        # A3
        # =========
        if use_memory and len(self.memory) > 0:
            z_mem, h_mem, reliability = self.memory.retrieve(c_t)
        else:
            z_mem, h_mem, reliability = None, None, 0.0

        # =========
        # A4
        # =========
        z_fused = self.fuse_affect(z_need, z_mem, reliability)     # z_t
        h_aff = self.hints_from_affect(z_fused)                    # h^aff_t
        h_fused = self.fuse_hints(h_need, h_mem, h_aff, reliability)  # h_t

        # arousal used for temperature schedules: canonical z[4] is arousal
        arousal = float(z_fused[4])
        q_pi = self.policy_probabilities(h_fused, arousal)

        # =========
        # A5
        # =========
        s_u, policy_scores_mat = self.compute_action_scores(q_pi, y_t)
        action_index = self.select_action(s_u, arousal)

        # Responsible policy tag (one-hot) for memory credit assignment:
        # pi_hat = argmax_pi q(pi) * s~_pi(u_t)
        contrib = q_pi * policy_scores_mat[:, action_index]
        pi_hat = int(np.argmax(contrib))
        policy_tag = one_hot(pi_hat, int(self.cfg["num_policies"]))

        # =========
        # A6
        # =========
        self.execute_action(action_index)

        # =========
        # A7
        # =========
        obs_post = self.observe()
        n_post = self.assess_needs(obs_post)
        affect_post_need = affect_from_needs(n_post, self.cfg)
        z_post, succ = self.compute_success(affect_need, affect_post_need)

        crossed_into_harm = bool(obs_post["in_harm_zone"]) and (not was_in_harm)

        # =========
        # A8
        # =========
        if use_memory:
            # Store either one-hot policy tag (recommended) or a normalized hint vector
            if bool(self.cfg.get("memory_store_one_hot_policy_tag", True)):
                stored_tag = policy_tag
            else:
                # fallback: store a clipped version of the fused hints as "tag"
                stored_tag = clip11(h_fused)

            ep = Episode(
                category=c_t.copy(),
                affect_fused=z_fused.copy(),
                hints_fused=h_fused.copy(),
                policy_tag=stored_tag.copy(),
                affect_post=z_post.copy(),
                success_post=float(succ),
            )
            stored = self.memory.store(ep)
        else:
            stored = False

        debug = {
            "num_episodes": len(self.memory),
            "stored": stored,
            "success": float(succ),
            "reliability": float(reliability),
            "policy_probs": q_pi.copy(),
            "pi_hat": pi_hat,
            "action_index": int(action_index),
        }
        return crossed_into_harm, int(np.argmax(q_pi)), debug


# ============================================================================
# WORLD SIMULATION
# ============================================================================

class World:
    def __init__(self, cfg: Dict, use_memory: bool = True):
        self.cfg = cfg
        self.use_memory = bool(use_memory)
        self.num_agents = int(cfg["num_agents"])
        self.time_step = 0
        self._init_agents()

        self.history = {
            "positions": [],
            "velocities": [],
            "crossings": [],
            "policies": [],
            "mean_episodes": [],
            "mean_y": [],
        }

    def _init_agents(self) -> None:
        self.agents: List[Agent] = []
        bx, by = float(self.cfg["world_bound_x"]), float(self.cfg["world_bound_y"])
        for _ in range(self.num_agents):
            pos_x = random.uniform(-bx * 0.8, bx * 0.8)
            pos_y = random.uniform(-by * 0.8, -by * 0.2)
            speed = random.uniform(0.5, float(self.cfg["target_speed"]))
            angle = random.uniform(-math.pi, math.pi)
            vel_x, vel_y = speed * math.cos(angle), speed * math.sin(angle)
            self.agents.append(Agent(pos_x, pos_y, vel_x, vel_y, self.cfg))

    def step(self) -> None:
        crossings = 0
        policies: List[int] = []
        num_episodes: List[int] = []

        for agent in self.agents:
            crossed, policy, dbg = agent.step(use_memory=self.use_memory)
            if crossed:
                crossings += 1
            policies.append(policy)
            num_episodes.append(int(dbg["num_episodes"]))

        self.history["positions"].append([(a.pos_x, a.pos_y) for a in self.agents])
        self.history["velocities"].append([(a.vel_x, a.vel_y) for a in self.agents])
        self.history["crossings"].append(int(crossings))
        self.history["policies"].append(policies)
        self.history["mean_episodes"].append(float(np.mean(num_episodes)) if num_episodes else 0.0)
        self.history["mean_y"].append(float(np.mean([a.pos_y for a in self.agents])))

        self.time_step += 1

    def run(self, num_steps: int, verbose: bool = True) -> None:
        for t in range(int(num_steps)):
            self.step()
            if verbose and (t % 100 == 0):
                mode = "MEM" if self.use_memory else "NO_MEM"
                print(f"[{mode}] t={t:4d}: crossings={sum(self.history['crossings']):4d}, "
                      f"mean_y={self.history['mean_y'][-1]:+6.2f}, "
                      f"mem={self.history['mean_episodes'][-1]:5.1f}")


# ============================================================================
# VISUALIZATION (unchanged style; optional)
# ============================================================================

def create_bird_marker(px, py, vx, vy, size=0.4):
    speed = math.sqrt(vx**2 + vy**2)
    angle = math.atan2(vy, vx) if speed > 0.01 else 0.0
    points = np.array([[size, 0], [-size*0.5, size*0.4], [-size*0.3, 0], [-size*0.5, -size*0.4]])
    c, s = math.cos(angle), math.sin(angle)
    rotated = points @ np.array([[c, s], [-s, c]])
    return Polygon(rotated + [px, py], closed=True)


def create_animation(world: World, filepath: str, fps: int = 20, skip: int = 4) -> None:
    print(f"Creating {filepath}...")
    fig, ax = plt.subplots(figsize=(12, 7))
    bx, by = float(world.cfg["world_bound_x"]), float(world.cfg["world_bound_y"])
    nb = int(world.cfg["num_y_bands"])

    ax.set_xlim(-bx*1.05, bx*1.05)
    ax.set_ylim(-by*1.15, by*1.15)
    ax.set_aspect("equal")

    ax.add_patch(Rectangle((-bx, 0), 2*bx, by, alpha=0.2, color="red"))
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)

    bh = 2*by/nb
    for i in range(1, nb):
        ax.axhline(y=-by+i*bh, color="blue", linestyle=":", alpha=0.3)

    ax.text(bx*0.85, by*0.7, "HARM", fontsize=12, color="red", fontweight="bold", alpha=0.7, ha="center")
    ax.text(bx*0.85, -by*0.7, "SAFE", fontsize=12, color="blue", fontweight="bold", alpha=0.6, ha="center")

    mode = "WITH MEMORY" if world.use_memory else "NO MEMORY"
    ax.text(0, by*1.05, mode, fontsize=14, ha="center", fontweight="bold",
            color="purple" if world.use_memory else "gray")

    frames = list(range(0, len(world.history["positions"]), skip))
    birds = []
    title = ax.set_title("")

    def update(frame_num):
        nonlocal birds
        for b in birds:
            b.remove()
        birds = []

        t = frames[frame_num]
        positions = world.history["positions"][t]
        velocities = world.history["velocities"][t]

        for (px, py), (vx, vy) in zip(positions, velocities):
            color = "red" if py > 0 else "green"
            bird = create_bird_marker(px, py, vx, vy, 0.5)
            bird.set_facecolor(color)
            bird.set_edgecolor("black")
            bird.set_linewidth(0.5)
            bird.set_alpha(0.8)
            ax.add_patch(bird)
            birds.append(bird)

        total = sum(world.history["crossings"][:t+1])
        recent = sum(world.history["crossings"][max(0, t-30):t+1])
        title.set_text(f"t={t}  crossings={total}  recent30={recent}  mean_y={world.history['mean_y'][t]:+.2f}")
        return birds + [title]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(filepath, writer=FFMpegWriter(fps=fps))
    plt.close()
    print(f"Saved {filepath}")


def plot_comparison(w_mem: World, w_no: World, filepath: str = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(np.cumsum(w_no.history["crossings"]), "r-", label="No Memory", lw=2)
    ax.plot(np.cumsum(w_mem.history["crossings"]), "b-", label="With Memory", lw=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Cumulative Crossings")
    ax.set_title("Total Crossings into Harm"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(w_no.history["mean_y"], "r-", label="No Memory", alpha=0.7)
    ax.plot(w_mem.history["mean_y"], "b-", label="With Memory", alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step"); ax.set_ylabel("Mean Y")
    ax.set_title("Vertical Position"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(w_mem.history["mean_episodes"], "b-", lw=2)
    ax.set_xlabel("Step"); ax.set_ylabel("Mean Episodes")
    ax.set_title("Memory Size"); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    T = len(w_mem.history["crossings"])
    T10 = max(1, T // 10)
    early_no, late_no = sum(w_no.history["crossings"][:T10]), sum(w_no.history["crossings"][-T10:])
    early_mem, late_mem = sum(w_mem.history["crossings"][:T10]), sum(w_mem.history["crossings"][-T10:])

    x = [0, 1]
    ax.bar([p-0.175 for p in x], [early_no, late_no], 0.35, label="No Memory", color="red", alpha=0.7)
    ax.bar([p+0.175 for p in x], [early_mem, late_mem], 0.35, label="With Memory", color="blue", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["Early", "Late"])
    ax.set_ylabel("Crossings")
    ax.set_title(f"Early vs Late\nNo:{early_no}→{late_no}  Mem:{early_mem}→{late_mem}")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Affect-Based Episodic Memory: A1-A8", fontsize=13)
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=150)
        print(f"Saved {filepath}")
    plt.close()


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment(cfg: Dict, seeds: List[int], num_steps: int = 2000) -> Dict:
    results = {"no_mem": [], "mem": [], "no_early": [], "mem_early": [], "no_late": [], "mem_late": []}

    for i, seed in enumerate(seeds):
        print(f"\n{'='*50}\nSeed {seed} ({i+1}/{len(seeds)})\n{'='*50}")

        random.seed(seed); np.random.seed(seed)
        w_no = World(cfg, use_memory=False)
        w_no.run(num_steps, verbose=True)

        random.seed(seed); np.random.seed(seed)
        w_mem = World(cfg, use_memory=True)
        w_mem.run(num_steps, verbose=True)

        T = len(w_mem.history["crossings"])
        T10 = max(1, T // 10)

        results["no_mem"].append(sum(w_no.history["crossings"]))
        results["mem"].append(sum(w_mem.history["crossings"]))
        results["no_early"].append(sum(w_no.history["crossings"][:T10]))
        results["mem_early"].append(sum(w_mem.history["crossings"][:T10]))
        results["no_late"].append(sum(w_no.history["crossings"][-T10:]))
        results["mem_late"].append(sum(w_mem.history["crossings"][-T10:]))

        print(f"No Mem: {results['no_mem'][-1]} (early={results['no_early'][-1]}, late={results['no_late'][-1]})")
        print(f"Mem:    {results['mem'][-1]} (early={results['mem_early'][-1]}, late={results['mem_late'][-1]})")

        if i == 0:
            import os
            os.makedirs("outputs", exist_ok=True)
            plot_comparison(w_mem, w_no, "outputs/harm-comparison.png")
            create_animation(w_no, "outputs/harm-no_memory.mp4", skip=3)
            create_animation(w_mem, "outputs/harm-with_memory.mp4", skip=3)

    return results


def print_summary(results: Dict) -> None:
    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    print(f"Total: No Mem={np.mean(results['no_mem']):.1f}±{np.std(results['no_mem']):.1f}, "
          f"Mem={np.mean(results['mem']):.1f}±{np.std(results['mem']):.1f}")
    print(f"Early: No Mem={np.mean(results['no_early']):.1f}, Mem={np.mean(results['mem_early']):.1f}")
    print(f"Late:  No Mem={np.mean(results['no_late']):.1f}, Mem={np.mean(results['mem_late']):.1f}")
    if np.mean(results["no_mem"]) > 0:
        red = (np.mean(results["no_mem"]) - np.mean(results["mem"])) / np.mean(results["no_mem"]) * 100.0
        print(f"Reduction: {red:+.1f}%")


if __name__ == "__main__":
    print("="*60 + "\nAFFECT-BASED EPISODIC MEMORY DEMO\nAlgorithm A1-A8\n" + "="*60)
    cfg = DEFAULT_CONFIG.copy()
    print("\nNeeds: Safety (harm at y>0), Motion (target speed)")
    print("Policies: Flee (south), Thrust (forward)")
    print("Memory: one-hot policy tags + normalized hints + success measures\n")

    seeds = [42, 123, 456, 789, 1011]
    results = run_experiment(cfg, seeds, num_steps=2500)
    print_summary(results)
    print("\nDone!")
