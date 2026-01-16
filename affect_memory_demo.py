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

Full implementation of Algorithm A1-A8 with SI-consistent normalization,
bounded hint fusion, and configurable success measures.

================================================================================
SCENARIO
================================================================================

Agents explore a 2D world with a harm zone (y > 0) that delivers negative 
affect ("electric shock") upon contact. No perceptual warning signal exists—
agents must learn from experience.

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

================================================================================
NORMALIZATION STRATEGY
================================================================================

1. H-Matrix Normalization:
   - H_need and H_affect matrices are row-L1 normalized at initialization
   - This ensures hints h = H @ x remain bounded in [-1, 1] when inputs 
     are in [-1, 1]

2. Hint Fusion Weights:
   - Base weights (alpha_need, alpha_mem, alpha_aff) are always renormalized 
     to sum to 1, ensuring fused hints stay bounded
   - When memory is absent, only alpha_need and alpha_aff contribute
   - Optional reliability weighting: when enabled, alpha_mem is scaled by
     retrieval reliability before renormalization (see design note below)

3. Memory Retrieval:
   - Uses simplified formula: h_mem = Σ w_j * succ_j * h_j
   - No denominator normalization, so success magnitude scales hint 
     strength linearly
   - This preserves success information for K=1 retrieval

================================================================================
DESIGN NOTE: RELIABILITY WEIGHTING
================================================================================

The `use_reliability_weighting` flag controls whether retrieval similarity
affects fusion weights:

    use_reliability_weighting = False (RECOMMENDED):
        - Similarity is used ONLY in retrieval (via w_j = softmax(sim))
        - Fusion uses fixed weights: no double-counting of similarity
        - Memory can still abstain if max_similarity < min_similarity threshold

    use_reliability_weighting = True:
        - Similarity affects BOTH retrieval weights AND fusion weights
        - alpha_mem is scaled by reliability before renormalization
        - This double-counts similarity, which may over-attenuate memory

The False setting is cleaner: let the retrieval weights w_j handle similarity,
and use fixed fusion weights. Memory naturally becomes weak when no good 
matches exist (w_j becomes diffuse, h_mem becomes averaged/weak).

================================================================================
CREDIT ASSIGNMENT
================================================================================

One-hot policy tagging isolates credit to the responsible policy:
    - At action selection, identify π_hat = argmax_π [q(π) * s̃_π(u_t)]
    - Store only that policy's one-hot tag with the episode
    - This prevents spurious credit leakage across policies

================================================================================
SUCCESS MEASURES (SI Equations)
================================================================================

Drive reduction (homeostatic):
    succ_drive = (||d||₁ - ||d*||₁) / max(||d||₁, ||d*||₁, ε)

Weighted emotion (hedonic):
    succ_i = (v*_i m*_i - v_i m_i) / (2 * max(|v*_i m*_i|, |v_i m_i|, ε))
    succ_emotion = Σᵢ wᵢ * succ_i / Σᵢ wᵢ

Hybrid:
    succ = ω * succ_drive + (1-ω) * succ_emotion

IMPORTANT: Success is computed from NEED-BASED affect (z^need, z*^need), 
not fused affect. This prevents "fear relief" artifacts when memory injects 
anticipatory valence into z_t.

================================================================================
ALGORITHM MAPPING (A1-A8)
================================================================================

A1: x_t ← observe()
    (c_t, y_t) ← categorize(x_t)

A2: n_t ← assess_needs(x_t)
    z^need_t ← affect_from_needs(n_t)
    h^need_t ← H_need @ d_t

A3: (z^mem_t, h^mem_t, reliability) ← memory.retrieve(c_t)

A4: z_t ← fuse_affect(z^need_t, z^mem_t)
    h^aff_t ← H_affect @ z_t
    h_t ← fuse_hints(h^need_t, h^mem_t, h^aff_t)
    q_t(π) ← softmax(h_t / τ₁(a_t))

A5: s_t(u) ← Σ_π q_t(π) * s̃_π(u)
    u_t ← select(s_t, τ₂(a_t))

A6: execute(u_t)

A7: x*_t ← observe()
    n*_t ← assess_needs(x*_t)
    z*_t ← affect_from_needs(n*_t)
    succ*_t ← compute_success(z^need_t, z*_t)

A8: if |succ*_t| ≥ threshold:
        store episode(c_t, z_t, h_t, π_hat, z*_t, succ*_t)

================================================================================
PHYSICS MODEL
================================================================================

State: (pos_x, pos_y, vel_x, vel_y) in bounded 2D arena
Actions: discrete acceleration directions (8 compass points) plus coast
Dynamics: acceleration with noise, velocity drag, boundary reflection

================================================================================
NEEDS AND POLICIES
================================================================================

Needs (2 channels):
    - Safety: n_safety = 1.0 in safe zone (y ≤ 0), 0.0 in harm zone (y > 0)
    - Motion: n_motion = speed / target_speed, clamped to [0, 1]

Policies (2):
    - Flee: accelerate south (innate safe direction)
    - Thrust: accelerate along current velocity (maintain motion)

Policy-to-action mappings are INNATE. The system learns WHEN to activate 
policies, not HOW to execute them.

================================================================================
EXPECTED RESULTS
================================================================================

    - Early phase: Similar crossing rates (learning period)
    - Late phase: close to ~0 crossings with memory vs. continued crossings without
    - Reduction: ~80% fewer total crossings with memory enabled

================================================================================
USAGE
================================================================================

    python affect_memory_demo.py

Outputs (in outputs/ directory):
    - harm-comparison.png: Side-by-side metrics
    - harm-no_memory.mp4: Animation without episodic memory
    - harm-with_memory.mp4: Animation with episodic memory

================================================================================
REFERENCE
================================================================================

"Synthetic Emotions and Consciousness: Exploring Architectural Boundaries"
See paper Figure 1 and Supplementary Information (SI) Part I.

Repository:
    https://github.com/affect-based-control/synthetic-emotion-controller

================================================================================
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


# ============================================================================
# CONSTANTS
# ============================================================================

EPS = 1e-12  # Small constant to prevent division by zero


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    """Clip a scalar value to the range [lo, hi]."""
    return max(lo, min(hi, x))


def clip01(arr: np.ndarray) -> np.ndarray:
    """Clip array elements to [0, 1]."""
    return np.clip(arr, 0.0, 1.0)


def clip11(arr: np.ndarray) -> np.ndarray:
    """Clip array elements to [-1, 1]."""
    return np.clip(arr, -1.0, 1.0)


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Compute numerically stable softmax with temperature scaling.
    
    Args:
        values: Input logits (any real values)
        temperature: Controls distribution sharpness
            - Lower temperature → more peaked (deterministic)
            - Higher temperature → flatter (exploratory)
    
    Returns:
        Probability distribution summing to 1
    """
    values = np.asarray(values, dtype=float)
    temperature = max(float(temperature), EPS)
    
    # Subtract max for numerical stability (prevents overflow)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted / temperature)
    
    return exp_values / (exp_values.sum() + EPS)


def l2_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    L2-normalize a vector to unit length.
    
    Returns zero vector if input norm is below eps threshold.
    This is used for category keys so that dot product equals cosine similarity.
    """
    x = np.asarray(x, dtype=float).flatten()
    norm = float(np.linalg.norm(x))
    
    if norm < eps:
        return np.zeros_like(x)
    
    return x / (norm + eps)


def normalize_rows_l1(M: np.ndarray) -> np.ndarray:
    """
    Row-wise L1 normalization of a matrix.
    
    After normalization, if input vector x has components in [-1, 1],
    then output h = M @ x will also have components in [-1, 1].
    
    This is essential for keeping hints bounded.
    """
    M = np.asarray(M, dtype=float)
    row_norms = np.sum(np.abs(M), axis=1, keepdims=True) + EPS
    return M / row_norms


def normalize_weights(*weights: float) -> Tuple[float, ...]:
    """
    Normalize a set of weights to sum to 1.
    
    This ensures that convex combinations of bounded quantities
    remain bounded. Used for hint fusion and affect fusion.
    
    Args:
        *weights: Variable number of non-negative weights
    
    Returns:
        Tuple of weights summing to 1
    """
    total = sum(weights) + EPS
    return tuple(w / total for w in weights)


def one_hot(index: int, size: int) -> np.ndarray:
    """
    Create a one-hot vector.
    
    Args:
        index: Position of the 1 (0-indexed)
        size: Length of the vector
    
    Returns:
        Vector of zeros with a single 1 at position index
    """
    vec = np.zeros(size, dtype=float)
    vec[int(index)] = 1.0
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Returns value in [-1, 1]:
        +1 = identical direction
         0 = orthogonal
        -1 = opposite direction
    """
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    
    norm_a = float(np.linalg.norm(a)) + EPS
    norm_b = float(np.linalg.norm(b)) + EPS
    
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG: Dict = {
    # -------------------------------------------------------------------------
    # World and Population
    # -------------------------------------------------------------------------
    "num_agents": 30,           # Number of agents in simulation
    "world_bound_x": 20.0,      # Half-width of arena (x ∈ [-bound, +bound])
    "world_bound_y": 10.0,      # Half-height of arena (y ∈ [-bound, +bound])
    "num_y_bands": 6,           # Number of categorical y-bands for memory
    "time_step": 0.2,           # Simulation time step (seconds)

    # -------------------------------------------------------------------------
    # Agent Motion Dynamics
    # -------------------------------------------------------------------------
    "max_speed": 3.0,           # Maximum allowed speed
    "target_speed": 1.5,        # Desired cruising speed (for motion need)
    "accel_magnitude": 2.0,     # Magnitude of acceleration actions
    "accel_noise_std": 0.4,     # Gaussian noise added to accelerations
    "velocity_drag": 0.03,      # Velocity decay per step (1 - drag)

    # -------------------------------------------------------------------------
    # Action Space
    # -------------------------------------------------------------------------
    "num_accel_directions": 8,  # Discrete acceleration directions (+ coast)

    # -------------------------------------------------------------------------
    # Policies
    # -------------------------------------------------------------------------
    "policy_names": ["Flee", "Thrust"],
    "num_policies": 2,
    "POLICY_FLEE": 0,           # Index for Flee policy
    "POLICY_THRUST": 1,         # Index for Thrust policy

    # -------------------------------------------------------------------------
    # Need System
    # -------------------------------------------------------------------------
    # Need targets n◇ (desired fulfillment levels)
    "need_target_safety": 0.9,  # Target safety fulfillment
    "need_target_motion": 0.6,  # Target motion fulfillment
    
    # Drive gains α (sensitivity of drives to need deviation)
    # d_i = tanh(α_i × (n◇_i - n_i))
    "drive_gain_safety": 4.0,   # High gain → strong drive when unsafe
    "drive_gain_motion": 2.5,   # Moderate gain for motion

    # -------------------------------------------------------------------------
    # Affect Computation
    # -------------------------------------------------------------------------
    # Valence scaling λ: v = -λ ⊙ d
    # Keep λ ≤ 1 to preserve bounds without clipping
    "valence_lambda": [1.0, 1.0],
    
    # Arousal model: a = base + scale × mean(|d|)
    "arousal_base": 0.3,        # Baseline arousal
    "arousal_scale": 0.5,       # How much drives increase arousal

    # -------------------------------------------------------------------------
    # Hint Matrices (will be row-L1 normalized at runtime)
    # -------------------------------------------------------------------------
    # H_need: [num_policies × num_needs] applied to drives d ∈ [-1,1]²
    # Positive entry (i,j) means: high drive j → favor policy i
    "H_need": [
        [ 2.0, -0.2],   # Flee: triggered by safety drive, slightly inhibited by motion drive
        [-0.3,  1.5],   # Thrust: triggered by motion drive
    ],
    
    # H_affect: [num_policies × affect_dim] applied to affect vector
    # Supported affect_dim values:
    #   3: [v_safety, v_motion, arousal]
    #   5: [v_safety, v_motion, arousal, d_safety, d_motion]
    #   7: [v_safety, v_motion, m_safety, m_motion, arousal, d_safety, d_motion]
    "H_affect": [
        [-2.0, -0.1,  0.3],   # Flee: negative valence → flee
        [ 0.2,  1.0, -0.1],   # Thrust: positive motion valence → thrust
    ],

    # -------------------------------------------------------------------------
    # Hint Fusion Weights
    # -------------------------------------------------------------------------
    # Base weights for combining hint sources (will be renormalized to sum to 1)
    "alpha_hint_need": 0.30,    # Weight for need-based hints h^need
    "alpha_hint_memory": 0.50,  # Weight for memory-based hints h^mem
    "alpha_hint_affect": 0.20,  # Weight for affect-based hints h^aff

    # -------------------------------------------------------------------------
    # Affect Fusion
    # -------------------------------------------------------------------------
    # Base weight for need-based affect in fusion: z = β×z_need + (1-β)×z_mem
    "beta_affect_fusion": 0.5,

    # -------------------------------------------------------------------------
    # Reliability Weighting (see design note in module docstring)
    # -------------------------------------------------------------------------
    # When True: alpha_mem is scaled by retrieval reliability before fusion
    # When False: fixed weights are used
    "use_reliability_weighting": False,

    # -------------------------------------------------------------------------
    # Temperature Schedules
    # -------------------------------------------------------------------------
    # Policy temperature τ₁(a): controls policy selection sharpness
    # τ₁ = max(base - arousal_scale × a, min)
    # Higher arousal → lower temperature → more decisive
    "tau_policy_base": 0.40,
    "tau_policy_arousal_scale": 0.25,
    "tau_policy_min": 0.05,
    
    # Action temperature τ₂(a): controls action selection sharpness
    "tau_action_base": 0.35,
    "tau_action_arousal_scale": 0.20,
    "tau_action_min": 0.05,

    # -------------------------------------------------------------------------
    # Policy Templates
    # -------------------------------------------------------------------------
    "flee_direction": -math.pi / 2,     # South (negative y direction)
    "policy_template_sharpness": 3.0,   # κ in exp(κ × (alignment - 1))

    # -------------------------------------------------------------------------
    # Episodic Memory System
    # -------------------------------------------------------------------------
    "memory_capacity": 100,             # Maximum stored episodes
    "memory_retrieval_k": 1,            # Number of neighbors to retrieve (k-NN)
    "memory_retrieval_tau": 1.0,        # Temperature for retrieval weight softmax
    "memory_min_similarity": 0.0,       # Abstain if max_sim < threshold
    "memory_tag_alpha": 0.15,           # Blend: z_tag = α×z_pre + (1-α)×z_post
    "memory_store_threshold": 0.25,     # Store only if |succ*| ≥ threshold

    # -------------------------------------------------------------------------
    # Credit Assignment
    # -------------------------------------------------------------------------
    # When True: store one-hot tag for responsible policy (recommended)
    # When False: store full hint vector (allows partial credit)
    "memory_store_one_hot_policy_tag": True,

    # -------------------------------------------------------------------------
    # Category Design
    # -------------------------------------------------------------------------
    # Additional category features reduce similarity ties for K=1 retrieval
    "category_include_vy_sign": True,   # Include velocity-y sign (approaching vs retreating)
    "category_include_harm_flag": True, # Include harm zone flag

    # -------------------------------------------------------------------------
    # Success Computation
    # -------------------------------------------------------------------------
    # Channel weights for hedonic success aggregation
    "success_weight_safety": 4.0,       # Prioritize safety outcomes
    "success_weight_motion": 1.0,
    
    # Success mode: "drive", "emotion", or "hybrid"
    "success_mode": "hybrid",
    "success_omega": 0.50,              # ω in hybrid: ω×drive + (1-ω)×emotion

    # -------------------------------------------------------------------------
    # Action Selection
    # -------------------------------------------------------------------------
    # Probability of using argmax vs sampling from softmax
    "action_use_argmax_probability": 0.80,
}


# ============================================================================
# AFFECT COMPUTATION
# ============================================================================

def affect_from_needs(needs: np.ndarray, cfg: Dict) -> Dict[str, np.ndarray]:
    """
    Compute affect state from need fulfillment levels.
    
    This implements the core affect model from the SI:
        d = tanh(α × (n◇ - n))     drives in [-1,1]
        v = -λ ⊙ d                 valence in [-1,1]
        m = |d|                    magnitude in [0,1]
        a = base + scale × mean(m) arousal in [0,1]
    
    Args:
        needs: Array [n_safety, n_motion] with values in [0,1]
        cfg: Configuration dictionary
    
    Returns:
        Dictionary containing:
            - "n": need levels [n_safety, n_motion]
            - "d": drives [d_safety, d_motion] in [-1,1]
            - "v": valence [v_safety, v_motion] in [-1,1]
            - "m": magnitude [m_safety, m_motion] in [0,1]
            - "a": arousal (scalar) in [0,1]
            - "z": canonical affect vector (length 7)
    
    Canonical z format:
        z = [v_safety, v_motion, m_safety, m_motion, arousal, d_safety, d_motion]
    """
    needs = np.asarray(needs, dtype=float).flatten()
    if needs.shape[0] != 2:
        raise ValueError(f"needs must have length 2, got {needs.shape[0]}")

    # Extract need levels
    n_safety, n_motion = float(needs[0]), float(needs[1])
    
    # Get targets and gains from config
    n_target = np.array([
        cfg["need_target_safety"], 
        cfg["need_target_motion"]
    ], dtype=float)
    
    alpha = np.array([
        cfg["drive_gain_safety"], 
        cfg["drive_gain_motion"]
    ], dtype=float)
    
    # Compute drives: positive = deficit (want more), negative = surplus
    # d = tanh(α × (n◇ - n))
    d = np.tanh(alpha * (n_target - needs))  # in [-1, 1]

    # Compute valence: v = -λ ⊙ d
    # Positive valence = feeling good = need satisfied = negative drive
    lam = np.array(cfg["valence_lambda"], dtype=float)
    v = clip11(-lam * d)

    # Compute magnitude: m = |d|
    m = np.abs(d)  # in [0, 1]

    # Compute arousal: increases with drive magnitude
    # a = base + scale × mean(|d|)
    a = clip(
        cfg["arousal_base"] + cfg["arousal_scale"] * float(np.mean(m)), 
        0.0, 
        1.0
    )

    # Assemble canonical affect vector
    z = np.array([
        v[0], v[1],      # valence: v_safety, v_motion
        m[0], m[1],      # magnitude: m_safety, m_motion
        a,               # arousal
        d[0], d[1]       # drives: d_safety, d_motion
    ], dtype=float)
    z = clip11(z)

    return {
        "n": needs.copy(),
        "d": d,
        "v": v,
        "m": m,
        "a": np.array([a], dtype=float),
        "z": z,
    }


# ============================================================================
# SUCCESS MEASURES
# ============================================================================

def success_drive_reduction(d_pre: np.ndarray, d_post: np.ndarray, eps: float = EPS) -> float:
    """
    Compute homeostatic success based on drive reduction.
    
    Formula (from SI):
        succ_drive = (||d||₁ - ||d*||₁) / max(||d||₁, ||d*||₁, ε)
    
    Interpretation:
        +1 = complete drive elimination (perfect success)
         0 = no change in drive magnitude
        -1 = drives doubled (complete failure)
    
    Args:
        d_pre: Pre-action drives
        d_post: Post-action drives
        eps: Small constant for numerical stability
    
    Returns:
        Success value in [-1, 1]
    """
    d_pre = np.asarray(d_pre, dtype=float).flatten()
    d_post = np.asarray(d_post, dtype=float).flatten()
    
    # L1 norms (sum of absolute drive magnitudes)
    norm_pre = float(np.sum(np.abs(d_pre)))
    norm_post = float(np.sum(np.abs(d_post)))
    
    # Normalized difference (positive = improvement)
    denom = max(norm_pre, norm_post, eps)
    
    return clip((norm_pre - norm_post) / denom, -1.0, 1.0)


def success_weighted_emotion(
    v_pre: np.ndarray, m_pre: np.ndarray,
    v_post: np.ndarray, m_post: np.ndarray,
    w: np.ndarray, eps: float = EPS
) -> float:
    """
    Compute hedonic success based on weighted emotional change.
    
    Formula (from SI):
        Per-channel: succ_i = (v*_i × m*_i - v_i × m_i) / (2 × max(|v*_i × m*_i|, |v_i × m_i|, ε))
        Aggregate: succ_emotion = Σᵢ wᵢ × succ_i / Σᵢ wᵢ
    
    The product v × m represents "signed emotional intensity":
        - Positive = intense positive emotion
        - Negative = intense negative emotion
        - Zero = neutral (either low valence or low intensity)
    
    Args:
        v_pre, m_pre: Pre-action valence and magnitude
        v_post, m_post: Post-action valence and magnitude
        w: Channel weights (e.g., [4.0, 1.0] to prioritize safety)
        eps: Small constant for numerical stability
    
    Returns:
        Success value in [-1, 1]
    """
    v_pre = np.asarray(v_pre, dtype=float).flatten()
    m_pre = np.asarray(m_pre, dtype=float).flatten()
    v_post = np.asarray(v_post, dtype=float).flatten()
    m_post = np.asarray(m_post, dtype=float).flatten()
    w = np.asarray(w, dtype=float).flatten()

    # Validate shapes
    if not (v_pre.shape == m_pre.shape == v_post.shape == m_post.shape == w.shape):
        raise ValueError("All arrays must have the same shape")

    # Compute per-channel success
    succ_per_channel = np.zeros_like(v_pre, dtype=float)
    
    for i in range(len(v_pre)):
        # Signed emotional intensity (v × m)
        intensity_pre = float(v_pre[i] * m_pre[i])
        intensity_post = float(v_post[i] * m_post[i])
        
        # Normalized change (positive = emotional improvement)
        denom = 2.0 * max(abs(intensity_post), abs(intensity_pre), eps)
        succ_per_channel[i] = (intensity_post - intensity_pre) / denom

    # Weighted average
    weight_sum = float(np.sum(w)) + EPS
    succ = float(np.sum(w * succ_per_channel) / weight_sum)
    
    return clip(succ, -1.0, 1.0)


def success_hybrid(succ_drive: float, succ_emotion: float, omega: float) -> float:
    """
    Compute hybrid success as weighted combination of drive and emotion measures.
    
    Formula:
        succ = ω × succ_drive + (1-ω) × succ_emotion
    
    Args:
        succ_drive: Homeostatic success in [-1, 1]
        succ_emotion: Hedonic success in [-1, 1]
        omega: Weight for drive component in [0, 1]
    
    Returns:
        Combined success in [-1, 1]
    """
    omega = clip(float(omega), 0.0, 1.0)
    combined = omega * float(succ_drive) + (1.0 - omega) * float(succ_emotion)
    return clip(combined, -1.0, 1.0)


# ============================================================================
# EPISODE DATA STRUCTURE
# ============================================================================

@dataclass
class Episode:
    """
    Data structure for a single episodic memory entry.
    
    Stored after action execution (A8) when |succ_post| ≥ threshold.
    
    Attributes:
        category: Categorical situation key c_t (L2-normalized for dot-product similarity)
        affect_fused: Fused affect state z_t at decision time
        hints_fused: Fused policy hints h_t at decision time
        policy_tag: One-hot tag indicating responsible policy (for credit assignment)
        affect_post: Post-action affect z*_t (from need-based computation)
        success_post: Post-action success succ*_t in [-1, 1]
    """
    category: np.ndarray      # c_t (L2-normalized)
    affect_fused: np.ndarray  # z_t
    hints_fused: np.ndarray   # h_t
    policy_tag: np.ndarray    # One-hot or full hints (based on config)
    affect_post: np.ndarray   # z*_t
    success_post: float       # succ*_t in [-1, 1]


# ============================================================================
# EPISODIC MEMORY SYSTEM
# ============================================================================

class EpisodicMemory:
    """
    Episodic memory system implementing A3 (retrieval) and A8 (storage).
    
    Key features:
        - Similarity-based retrieval using L2-normalized category keys
        - Selective storage: only episodes with |succ*| ≥ threshold
        - Optional abstention when no similar episodes exist
        - Configurable k-NN retrieval with softmax weighting
    
    Memory formula (simplified from SI):
        h_mem = Σⱼ wⱼ × succ*ⱼ × hⱼ
        z_mem = Σⱼ wⱼ × z_tagⱼ
        
    where:
        wⱼ = softmax(τ × sim(c_t, c_j))
        z_tagⱼ = α × z_j + (1-α) × z*_j
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize episodic memory system.
        
        Args:
            cfg: Configuration dictionary
        """
        self.capacity = int(cfg["memory_capacity"])
        self.retrieval_k = int(cfg["memory_retrieval_k"])
        self.retrieval_tau = float(cfg.get("memory_retrieval_tau", 1.0))
        self.min_similarity = float(cfg.get("memory_min_similarity", 0.0))
        self.tag_alpha = float(cfg["memory_tag_alpha"])
        self.store_threshold = float(cfg["memory_store_threshold"])
        self.num_policies = int(cfg["num_policies"])
        
        # Bounded deque automatically removes oldest when full
        self.episodes: deque = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        """Return number of stored episodes."""
        return len(self.episodes)

    def store(self, episode: Episode) -> bool:
        """
        Store an episode if it meets the significance threshold.
        
        Implements A8: selective storage to prevent memory dilution
        from neutral experiences.
        
        Args:
            episode: Episode to potentially store
        
        Returns:
            True if episode was stored, False otherwise
        """
        # Only store significant episodes
        if abs(float(episode.success_post)) < self.store_threshold:
            return False

        # L2-normalize category key for dot-product similarity
        # This ensures sim(c_t, c_j) = c_t · c_j = cos(c_t, c_j)
        episode.category = l2_normalize(episode.category)
    
        self.episodes.append(episode)
        return True

    def retrieve(self, category_current: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Retrieve aggregated affect and hints from similar past episodes.
        
        Implements A3: episodic retrieval using k-NN with softmax weighting.
        
        Memory formulas:
            sim_j = c_t · c_j              (dot product of L2-normalized keys)
            w_j = softmax(τ × sim_j)       (retrieval weights, sum to 1)
            h_mem = Σⱼ wⱼ × succ*ⱼ × hⱼ   (success-weighted hints)
            z_mem = Σⱼ wⱼ × z_tagⱼ        (weighted affect)
            z_tagⱼ = α × zⱼ + (1-α) × z*ⱼ (blend pre/post affect)
        
        Args:
            category_current: Current situation category c_t
        
        Returns:
            Tuple of (z_mem, h_mem, reliability):
                - z_mem: Retrieved affect (None if empty memory)
                - h_mem: Retrieved hints (None if empty memory)
                - reliability: Max similarity score in [0, 1]
        """
        # Handle empty memory
        if len(self.episodes) == 0:
            return None, None, 0.0

        # L2-normalize query for dot-product similarity
        c_t = l2_normalize(category_current)
    
        # Compute similarities to all stored episodes
        # Since both c_t and c_j are L2-normalized, dot product = cosine similarity
        sims = np.array([
            float(np.dot(c_t, ep.category)) 
            for ep in self.episodes
        ], dtype=float)
    
        # Select top-K most similar episodes
        k = min(self.retrieval_k, len(self.episodes))
        top_indices = np.argsort(sims)[-k:]
        top_sims = sims[top_indices]
    
        # Reliability = max similarity (used for optional abstention and weighting)
        reliability = float(np.max(top_sims)) if top_sims.size else 0.0
        reliability = clip(reliability, 0.0, 1.0)
    
        # Optional abstention: if no sufficiently similar episode exists
        if reliability < self.min_similarity:
            z_dim = int(self.episodes[0].affect_fused.shape[0])
            return np.zeros(z_dim, dtype=float), np.zeros(self.num_policies, dtype=float), 0.0

        # Compute retrieval weights: w = softmax(τ × sim)
        w = softmax(self.retrieval_tau * top_sims, temperature=1.0)

        # Aggregate affect: z_mem = Σⱼ wⱼ × z_tagⱼ
        z_dim = int(self.episodes[0].affect_fused.shape[0])
        z_mem = np.zeros(z_dim, dtype=float)
        
        for idx, wi in zip(top_indices, w):
            ep = self.episodes[int(idx)]
            # Blend pre-action and post-action affect
            z_tag = self.tag_alpha * ep.affect_fused + (1.0 - self.tag_alpha) * ep.affect_post
            z_mem += float(wi) * z_tag
        
        z_mem = clip11(z_mem)

        # Aggregate hints: h_mem = Σⱼ wⱼ × succ*ⱼ × hⱼ
        # Note: no denominator normalization, so success magnitude scales linearly
        h_mem = np.zeros(self.num_policies, dtype=float)
        
        for idx, wi in zip(top_indices, w):
            ep = self.episodes[int(idx)]
            h_mem += float(wi) * float(ep.success_post) * ep.policy_tag
        
        h_mem = clip11(h_mem)

        return z_mem, h_mem, reliability


# ============================================================================
# AGENT
# ============================================================================

class Agent:
    """
    Agent implementing the full A1-A8 emotion-like control loop.
    
    The agent:
        1. Observes the world and categorizes the situation (A1)
        2. Appraises needs and computes affect (A2)
        3. Retrieves similar episodes from memory (A3)
        4. Fuses affect and hints to select policies (A4)
        5. Instantiates policies into action scores (A5)
        6. Executes the selected action (A6)
        7. Reappraises the outcome (A7)
        8. Stores significant episodes (A8)
    """
    
    def __init__(self, pos_x: float, pos_y: float, vel_x: float, vel_y: float, cfg: Dict):
        """
        Initialize an agent at a given position and velocity.
        
        Args:
            pos_x, pos_y: Initial position
            vel_x, vel_y: Initial velocity
            cfg: Configuration dictionary
        """
        # State
        self.pos_x = float(pos_x)
        self.pos_y = float(pos_y)
        self.vel_x = float(vel_x)
        self.vel_y = float(vel_y)
        self.cfg = cfg

        # Episodic memory
        self.memory = EpisodicMemory(cfg)
        
        # Action space
        self.actions = self._build_action_space()
        self.num_actions = len(self.actions)

        # Hint matrices (row-L1 normalized to keep hints bounded)
        self.H_need = normalize_rows_l1(np.array(cfg["H_need"], dtype=float))
        self.H_affect = normalize_rows_l1(np.array(cfg["H_affect"], dtype=float))

        # Validate dimensions
        if self.H_need.shape != (cfg["num_policies"], 2):
            raise ValueError(
                f"H_need shape must be ({cfg['num_policies']}, 2), "
                f"got {self.H_need.shape}"
            )

    def _build_action_space(self) -> List[Tuple[float, float]]:
        """
        Build the discrete action space.
        
        Actions are acceleration vectors in 2D:
            - num_accel_directions equally spaced directions
            - Plus a "coast" action (zero acceleration)
        
        Returns:
            List of (ax, ay) acceleration tuples
        """
        actions: List[Tuple[float, float]] = []
        magnitude = float(self.cfg["accel_magnitude"])
        num_directions = int(self.cfg["num_accel_directions"])
        
        # Directional accelerations
        for i in range(num_directions):
            angle = 2.0 * math.pi * i / num_directions
            actions.append((
                magnitude * math.cos(angle), 
                magnitude * math.sin(angle)
            ))
        
        # Coast action (no acceleration)
        actions.append((0.0, 0.0))
        
        return actions

    # =========================================================================
    # A1: OBSERVE AND CATEGORIZE
    # =========================================================================

    def observe(self) -> Dict:
        """
        A1 (line 1): Observe the current state from the world.
        
        Returns raw observations that will be used for:
            - Categorization (for memory retrieval)
            - Need assessment (for affect computation)
            - Policy instantiation (situation parameters)
        
        Returns:
            Dictionary with observation fields
        """
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
        """
        A1 (line 2): Categorize the observation into (c_t, y_t).
        
        The category c_t is a coarse representation used for memory retrieval.
        The parameters y_t are fine-grained values used for policy instantiation.
        
        Category design (to reduce similarity ties):
            c_t = [y_band one-hot, sign(v_y) one-hot, harm_flag one-hot]
        
        Args:
            obs: Raw observation dictionary
        
        Returns:
            Tuple of (category, parameters):
                - category: Categorical key for memory retrieval
                - parameters: Situation parameters for policy instantiation
        """
        # Y-band one-hot (which vertical region is the agent in?)
        num_bands = int(self.cfg["num_y_bands"])
        bound_y = float(self.cfg["world_bound_y"])
        band_height = 2.0 * bound_y / num_bands
        
        band_index = int((obs["pos_y"] + bound_y) / band_height)
        band_index = int(clip(band_index, 0, num_bands - 1))
        
        c_band = np.zeros(num_bands, dtype=float)
        c_band[band_index] = 1.0

        # Additional category features (reduce similarity ties)
        extra_features = []
        
        # Velocity-y sign (approaching vs retreating from boundary)
        if self.cfg.get("category_include_vy_sign", True):
            vy_positive = 1 if obs["vel_y"] > 0.0 else 0
            c_vy = np.zeros(2, dtype=float)
            c_vy[vy_positive] = 1.0
            extra_features.append(c_vy)

        # Harm zone flag (currently in harm vs safe)
        if self.cfg.get("category_include_harm_flag", True):
            in_harm = 1 if obs["pos_y"] > 0.0 else 0
            c_harm = np.zeros(2, dtype=float)
            c_harm[in_harm] = 1.0
            extra_features.append(c_harm)

        # Concatenate all features
        if extra_features:
            category = np.concatenate([c_band] + extra_features)
        else:
            category = c_band

        # Situation parameters for policy instantiation
        y_t = {
            "vel_x": float(obs["vel_x"]),
            "vel_y": float(obs["vel_y"]),
            "speed": float(obs["speed"]),
        }
        
        return category, y_t

    # =========================================================================
    # A2: NEED APPRAISAL
    # =========================================================================

    def assess_needs(self, obs: Dict) -> np.ndarray:
        """
        A2 (line 3): Assess current need fulfillment levels.
        
        Needs are first-order perceptual quantities derived directly
        from current observations (no metarepresentation).
        
        Args:
            obs: Raw observation dictionary
        
        Returns:
            Array [n_safety, n_motion] with values in [0, 1]
        """
        # Safety need: fulfilled when not in harm zone
        need_safety = 0.0 if obs["in_harm_zone"] else 1.0
        
        # Motion need: fulfilled when moving at target speed
        need_motion = clip(
            obs["speed"] / self.cfg["target_speed"], 
            0.0, 
            1.0
        )
        
        return np.array([need_safety, need_motion], dtype=float)

    def hints_from_needs(self, drives: np.ndarray) -> np.ndarray:
        """
        A2 (line 5): Compute need-based policy hints.
        
        Formula: h^need = H_need @ d
        
        Since H_need is row-L1 normalized and d ∈ [-1,1]²,
        the output is bounded in [-1,1].
        
        Args:
            drives: Drive vector d ∈ [-1,1]²
        
        Returns:
            Hint vector h^need ∈ [-1,1]^num_policies
        """
        h = self.H_need @ np.asarray(drives, dtype=float).flatten()
        return clip11(h)

    # =========================================================================
    # A4: AFFECT AND HINT FUSION
    # =========================================================================

    def fuse_affect(
        self, 
        z_need: np.ndarray, 
        z_mem: Optional[np.ndarray],
        reliability: float
    ) -> np.ndarray:
        """
        A4 (line 9): Fuse need-based and memory-based affect.
        
        Formula: z_t = w_need × z^need + w_mem × z^mem
        
        Weight computation depends on use_reliability_weighting flag:
            - False: Fixed weights (β, 1-β)
            - True: Reliability-modulated weights
        
        Weights are always normalized to sum to 1.
        
        Args:
            z_need: Need-based affect
            z_mem: Memory-based affect (None if memory absent)
            reliability: Retrieval reliability in [0, 1]
        
        Returns:
            Fused affect z_t
        """
        # If no memory, return need-based affect
        if z_mem is None:
            return clip11(np.asarray(z_need, dtype=float).copy())

        z_need = np.asarray(z_need, dtype=float).flatten()
        z_mem = np.asarray(z_mem, dtype=float).flatten()
        
        if z_need.shape != z_mem.shape:
            raise ValueError(
                f"z_need and z_mem must have same shape, "
                f"got {z_need.shape} vs {z_mem.shape}"
            )

        beta = float(self.cfg["beta_affect_fusion"])
        use_reliability = bool(self.cfg.get("use_reliability_weighting", False))
        
        if use_reliability:
            # Reliability-modulated weights
            # As reliability increases, trust memory more
            r = clip(float(reliability), 0.0, 1.0)
            w_need_raw = beta + (1.0 - beta) * (1.0 - r)
            w_mem_raw = (1.0 - beta) * r
        else:
            # Fixed weights (no reliability modulation)
            w_need_raw = beta
            w_mem_raw = 1.0 - beta
        
        # Normalize weights to sum to 1
        w_need, w_mem = normalize_weights(w_need_raw, w_mem_raw)

        # Fuse
        z = w_need * z_need + w_mem * z_mem
        return clip11(z)

    def _affect_input_for_H_affect(self, z_fused: np.ndarray) -> np.ndarray:
        """
        Extract the affect input vector for H_affect multiplication.
        
        Canonical z has length 7:
            [v_safety, v_motion, m_safety, m_motion, arousal, d_safety, d_motion]
        
        H_affect may expect different subsets:
            - 3 columns: [v_safety, v_motion, arousal]
            - 5 columns: [v_safety, v_motion, arousal, d_safety, d_motion]
            - 7 columns: full z
        
        Args:
            z_fused: Full fused affect vector (length 7)
        
        Returns:
            Affect input vector matching H_affect column count
        """
        z = np.asarray(z_fused, dtype=float).flatten()
        
        if z.shape[0] != 7:
            raise ValueError(f"Expected z length 7, got {z.shape[0]}")
        
        cols = int(self.H_affect.shape[1])

        if cols == 3:
            # [v_safety, v_motion, arousal]
            return np.array([z[0], z[1], z[4]], dtype=float)
        elif cols == 5:
            # [v_safety, v_motion, arousal, d_safety, d_motion]
            return np.array([z[0], z[1], z[4], z[5], z[6]], dtype=float)
        elif cols == 7:
            return z.copy()
        else:
            raise ValueError(
                f"Unsupported H_affect column count: {cols} (supported: 3, 5, 7)"
            )

    def hints_from_affect(self, z_fused: np.ndarray) -> np.ndarray:
        """
        A4 (line 10): Compute affect-based policy hints.
        
        Formula: h^aff = H_affect @ z (or projection of z)
        
        Args:
            z_fused: Fused affect vector
        
        Returns:
            Hint vector h^aff ∈ [-1,1]^num_policies
        """
        x = self._affect_input_for_H_affect(z_fused)
        h = self.H_affect @ x
        return clip11(h)

    def fuse_hints(
        self,
        h_need: np.ndarray,
        h_mem: Optional[np.ndarray],
        h_aff: np.ndarray,
        reliability: float
    ) -> np.ndarray:
        """
        A4 (line 11): Fuse hints from all sources.
        
        Formula: h_t = α_need × h^need + α_mem × h^mem + α_aff × h^aff
        
        Weights are always normalized to sum to 1, ensuring fused
        hints remain bounded in [-1, 1].
        
        Weight handling:
            - Memory absent: Use only need and affect weights
            - Memory present + use_reliability=False: Use fixed weights
            - Memory present + use_reliability=True: Scale α_mem by reliability
        
        Args:
            h_need: Need-based hints
            h_mem: Memory-based hints (None if memory absent)
            h_aff: Affect-based hints
            reliability: Retrieval reliability in [0, 1]
        
        Returns:
            Fused hint vector h_t ∈ [-1,1]^num_policies
        """
        h_need = clip11(np.asarray(h_need, dtype=float).flatten())
        h_aff = clip11(np.asarray(h_aff, dtype=float).flatten())

        # Get base weights from config
        alpha_need_base = float(self.cfg["alpha_hint_need"])
        alpha_mem_base = float(self.cfg["alpha_hint_memory"])
        alpha_aff_base = float(self.cfg["alpha_hint_affect"])
        use_reliability = bool(self.cfg.get("use_reliability_weighting", False))

        if h_mem is not None:
            # Memory is available
            h_mem_vec = clip11(np.asarray(h_mem, dtype=float).flatten())
            
            if use_reliability:
                # Scale memory weight by reliability
                r = clip(float(reliability), 0.0, 1.0)
                alpha_mem_raw = alpha_mem_base * r
            else:
                # Fixed memory weight (recommended)
                alpha_mem_raw = alpha_mem_base
            
            # Normalize all weights to sum to 1
            alpha_need, alpha_mem, alpha_aff = normalize_weights(
                alpha_need_base, alpha_mem_raw, alpha_aff_base
            )
            
            # Fuse all three sources
            h = alpha_need * h_need + alpha_mem * h_mem_vec + alpha_aff * h_aff
        else:
            # Memory absent: use only need and affect
            alpha_need, alpha_aff = normalize_weights(alpha_need_base, alpha_aff_base)
            h = alpha_need * h_need + alpha_aff * h_aff

        return clip11(h)

    def policy_probabilities(self, h_fused: np.ndarray, arousal: float) -> np.ndarray:
        """
        A4 (line 12): Convert fused hints to policy probabilities.
        
        Formula: q(π) = softmax(h_t / τ₁(a_t))
        
        Temperature schedule: τ₁ = max(base - scale × arousal, min)
        Higher arousal → lower temperature → more decisive selection
        
        Args:
            h_fused: Fused hint vector
            arousal: Current arousal level in [0, 1]
        
        Returns:
            Policy probability distribution summing to 1
        """
        a = clip(float(arousal), 0.0, 1.0)
        
        # Temperature decreases with arousal
        tau = max(
            self.cfg["tau_policy_base"] - self.cfg["tau_policy_arousal_scale"] * a,
            self.cfg["tau_policy_min"]
        )
        
        return softmax(np.asarray(h_fused, dtype=float), temperature=tau)

    # =========================================================================
    # A5: POLICY INSTANTIATION
    # =========================================================================

    def score_actions_for_policy(self, policy_index: int, y_t: Dict) -> np.ndarray:
        """
        A5 (line 13): Compute action scores for a specific policy.
        
        Policy templates define how abstract policies map to concrete actions
        given the current situation parameters.
        
        Policies:
            - Flee: Score actions by alignment with flee direction (south)
            - Thrust: Score actions by alignment with current velocity
        
        Args:
            policy_index: Which policy to instantiate
            y_t: Situation parameters (velocity, speed)
        
        Returns:
            Action score vector s̃_π(u) ∈ [0,1]^num_actions
        """
        scores = np.zeros(self.num_actions, dtype=float)
        kappa = float(self.cfg["policy_template_sharpness"])

        if policy_index == self.cfg["POLICY_FLEE"]:
            # Flee: prefer actions aligned with flee direction
            flee_dir = float(self.cfg["flee_direction"])
            flee_x, flee_y = math.cos(flee_dir), math.sin(flee_dir)

            for i, (ax, ay) in enumerate(self.actions):
                accel_norm = math.sqrt(ax**2 + ay**2)
                
                if accel_norm < EPS:
                    # Coast action: moderate score
                    scores[i] = 0.30
                else:
                    # Alignment in [-1, 1]
                    alignment = (ax * flee_x + ay * flee_y) / accel_norm
                    # Score via von Mises-like kernel
                    scores[i] = math.exp(kappa * (alignment - 1.0))

        elif policy_index == self.cfg["POLICY_THRUST"]:
            # Thrust: prefer actions aligned with current velocity
            vx, vy = float(y_t["vel_x"]), float(y_t["vel_y"])
            speed = float(y_t["speed"])

            if speed < 0.1:
                # Nearly stationary: any motion is good
                for i, (ax, ay) in enumerate(self.actions):
                    accel_norm = math.sqrt(ax**2 + ay**2)
                    scores[i] = 0.50 if accel_norm > 0.1 else 0.30
            else:
                # Moving: prefer actions aligned with velocity
                ux, uy = vx / speed, vy / speed
                
                for i, (ax, ay) in enumerate(self.actions):
                    accel_norm = math.sqrt(ax**2 + ay**2)
                    
                    if accel_norm < EPS:
                        scores[i] = 0.50
                    else:
                        alignment = (ax * ux + ay * uy) / accel_norm
                        # Softer kernel than Flee
                        scores[i] = math.exp(kappa * (alignment - 1.0) / 2.0)

        return clip01(scores)

    def compute_action_scores(
        self, 
        q_pi: np.ndarray, 
        y_t: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        A5 (line 14): Compute final action scores by mixing policy scores.
        
        Formula: s_t(u) = Σ_π q(π) × s̃_π(u)
        
        Args:
            q_pi: Policy probabilities
            y_t: Situation parameters
        
        Returns:
            Tuple of (action_scores, policy_scores_matrix):
                - action_scores: Final scores s_t(u) ∈ [0,1]^num_actions
                - policy_scores_matrix: Per-policy scores for credit assignment
        """
        q = np.asarray(q_pi, dtype=float).flatten()
        
        if q.shape[0] != self.cfg["num_policies"]:
            raise ValueError("Policy probability vector has wrong length")

        # Compute scores for each policy
        policy_scores = np.zeros((self.cfg["num_policies"], self.num_actions), dtype=float)
        
        for pi in range(self.cfg["num_policies"]):
            policy_scores[pi, :] = self.score_actions_for_policy(pi, y_t)

        # Mix policy scores according to policy probabilities
        action_scores = np.zeros(self.num_actions, dtype=float)
        for pi in range(self.cfg["num_policies"]):
            action_scores += float(q[pi]) * policy_scores[pi, :]

        return clip01(action_scores), clip01(policy_scores)

    def select_action(self, s_u: np.ndarray, arousal: float) -> int:
        """
        A5 (line 15): Select an action from the score distribution.
        
        Selection method:
            - With probability p: argmax (exploit)
            - With probability 1-p: sample from softmax (explore)
        
        Temperature schedule: τ₂ = max(base - scale × arousal, min)
        
        Args:
            s_u: Action scores
            arousal: Current arousal level
        
        Returns:
            Index of selected action
        """
        a = clip(float(arousal), 0.0, 1.0)
        
        # Temperature decreases with arousal
        tau = max(
            self.cfg["tau_action_base"] - self.cfg["tau_action_arousal_scale"] * a,
            self.cfg["tau_action_min"]
        )
        
        # Compute action probabilities
        probs = softmax(np.asarray(s_u, dtype=float), temperature=tau)

        # Select action
        if random.random() < self.cfg["action_use_argmax_probability"]:
            return int(np.argmax(s_u))
        else:
            return int(random.choices(range(self.num_actions), weights=probs)[0])

    # =========================================================================
    # A6: EXECUTE
    # =========================================================================

    def execute_action(self, action_index: int) -> None:
        """
        A6 (line 16): Execute the selected action in the world.
        
        Physics:
            1. Apply acceleration (with noise)
            2. Apply velocity drag
            3. Cap speed at maximum
            4. Update position
            5. Handle boundary reflections
        
        Args:
            action_index: Index of action to execute
        """
        ax, ay = self.actions[int(action_index)]
        dt = float(self.cfg["time_step"])

        # Add acceleration noise
        ax += random.gauss(0.0, self.cfg["accel_noise_std"])
        ay += random.gauss(0.0, self.cfg["accel_noise_std"])

        # Update velocity
        self.vel_x += ax * dt
        self.vel_y += ay * dt

        # Apply drag
        drag = float(self.cfg["velocity_drag"])
        self.vel_x *= (1.0 - drag)
        self.vel_y *= (1.0 - drag)

        # Cap speed
        speed = math.sqrt(self.vel_x**2 + self.vel_y**2)
        max_speed = float(self.cfg["max_speed"])
        
        if speed > max_speed:
            scale = max_speed / (speed + EPS)
            self.vel_x *= scale
            self.vel_y *= scale

        # Update position
        self.pos_x += self.vel_x * dt
        self.pos_y += self.vel_y * dt

        # Handle boundary reflections
        bx = float(self.cfg["world_bound_x"])
        by = float(self.cfg["world_bound_y"])
        margin = 0.1
        bounce = 0.8

        # X boundaries
        if self.pos_x > bx - margin:
            self.pos_x = bx - margin
            self.vel_x = -abs(self.vel_x) * bounce
        elif self.pos_x < -bx + margin:
            self.pos_x = -bx + margin
            self.vel_x = abs(self.vel_x) * bounce

        # Y boundaries
        if self.pos_y > by - margin:
            self.pos_y = by - margin
            self.vel_y = -abs(self.vel_y) * bounce
        elif self.pos_y < -by + margin:
            self.pos_y = -by + margin
            self.vel_y = abs(self.vel_y) * bounce

    # =========================================================================
    # A7: REAPPRAISE
    # =========================================================================

    def compute_success(
        self, 
        affect_pre_need: Dict, 
        affect_post_need: Dict
    ) -> Tuple[np.ndarray, float]:
        """
        A7 (line 18): Compute post-action affect and success.
        
        IMPORTANT: Success is computed from NEED-BASED affect, not fused
        affect. This prevents "fear relief" artifacts when memory injects
        anticipatory valence.
        
        Args:
            affect_pre_need: Pre-action need-based affect (from A2)
            affect_post_need: Post-action need-based affect
        
        Returns:
            Tuple of (z_post, success):
                - z_post: Post-action affect vector
                - success: Success value in [-1, 1]
        """
        # Extract components for success computation
        d_pre = affect_pre_need["d"]
        d_post = affect_post_need["d"]
        v_pre = affect_pre_need["v"]
        v_post = affect_post_need["v"]
        m_pre = affect_pre_need["m"]
        m_post = affect_post_need["m"]

        # Channel weights
        w = np.array([
            self.cfg["success_weight_safety"], 
            self.cfg["success_weight_motion"]
        ], dtype=float)

        # Compute success measures
        succ_drive = success_drive_reduction(d_pre, d_post)
        succ_emotion = success_weighted_emotion(v_pre, m_pre, v_post, m_post, w)

        # Select success measure based on config
        mode = str(self.cfg.get("success_mode", "hybrid")).lower()
        
        if mode == "drive":
            succ = succ_drive
        elif mode == "emotion":
            succ = succ_emotion
        elif mode == "hybrid":
            omega = float(self.cfg.get("success_omega", 0.5))
            succ = success_hybrid(succ_drive, succ_emotion, omega)
        else:
            raise ValueError(f"Unknown success_mode: {mode}")

        return affect_post_need["z"], clip(float(succ), -1.0, 1.0)

    # =========================================================================
    # FULL A1-A8 LOOP
    # =========================================================================

    def step(self, use_memory: bool = True) -> Tuple[bool, int, Dict]:
        """
        Execute one complete A1-A8 iteration.
        
        This is the main control loop implementing emotion-like control.
        
        Args:
            use_memory: Whether to use episodic memory (A3, A8)
        
        Returns:
            Tuple of (crossed_into_harm, policy_argmax, debug):
                - crossed_into_harm: True if agent entered harm zone this step
                - policy_argmax: Index of most probable policy
                - debug: Dictionary with internal state for analysis
        """
        # =====================================================================
        # A1: OBSERVE AND CATEGORIZE
        # =====================================================================
        obs = self.observe()
        c_t, y_t = self.categorize(obs)
        was_in_harm = obs["in_harm_zone"]

        # =====================================================================
        # A2: NEED APPRAISAL
        # =====================================================================
        n_t = self.assess_needs(obs)
        affect_need = affect_from_needs(n_t, self.cfg)
        z_need = affect_need["z"]
        h_need = self.hints_from_needs(affect_need["d"])

        # =====================================================================
        # A3: EPISODIC RETRIEVAL
        # =====================================================================
        if use_memory and len(self.memory) > 0:
            z_mem, h_mem, reliability = self.memory.retrieve(c_t)
        else:
            z_mem, h_mem, reliability = None, None, 0.0

        # =====================================================================
        # A4: AFFECTIVE INTEGRATION
        # =====================================================================
        # Fuse affect from needs and memory
        z_fused = self.fuse_affect(z_need, z_mem, reliability)
        
        # Compute affect-based hints
        h_aff = self.hints_from_affect(z_fused)
        
        # Fuse all hint sources
        h_fused = self.fuse_hints(h_need, h_mem, h_aff, reliability)

        # Convert hints to policy probabilities
        arousal = float(z_fused[4])  # Index 4 is arousal in canonical z
        q_pi = self.policy_probabilities(h_fused, arousal)

        # =====================================================================
        # A5: POLICY INSTANTIATION
        # =====================================================================
        # Compute action scores
        s_u, policy_scores_mat = self.compute_action_scores(q_pi, y_t)
        
        # Select action
        action_index = self.select_action(s_u, arousal)

        # Determine responsible policy for credit assignment
        # π_hat = argmax_π [q(π) × s̃_π(u_t)]
        contrib = q_pi * policy_scores_mat[:, action_index]
        pi_hat = int(np.argmax(contrib))
        policy_tag = one_hot(pi_hat, self.cfg["num_policies"])

        # =====================================================================
        # A6: EXECUTE
        # =====================================================================
        self.execute_action(action_index)

        # =====================================================================
        # A7: REAPPRAISE
        # =====================================================================
        obs_post = self.observe()
        n_post = self.assess_needs(obs_post)
        affect_post_need = affect_from_needs(n_post, self.cfg)
        z_post, succ = self.compute_success(affect_need, affect_post_need)

        # Check if crossed into harm zone
        crossed_into_harm = obs_post["in_harm_zone"] and not was_in_harm

        # =====================================================================
        # A8: EPISODE STORAGE
        # =====================================================================
        stored = False
        
        if use_memory:
            # Choose what to store as policy tag
            if self.cfg.get("memory_store_one_hot_policy_tag", True):
                stored_tag = policy_tag
            else:
                stored_tag = clip11(h_fused)

            # Create and store episode
            episode = Episode(
                category=c_t.copy(),
                affect_fused=z_fused.copy(),
                hints_fused=h_fused.copy(),
                policy_tag=stored_tag.copy(),
                affect_post=z_post.copy(),
                success_post=float(succ),
            )
            stored = self.memory.store(episode)

        # Prepare debug output
        debug = {
            "num_episodes": len(self.memory),
            "stored": stored,
            "success": float(succ),
            "reliability": float(reliability),
            "policy_probs": q_pi.copy(),
            "pi_hat": pi_hat,
            "action_index": int(action_index),
            "arousal": arousal,
            "z_fused": z_fused.copy(),
            "h_fused": h_fused.copy(),
        }
        
        return crossed_into_harm, int(np.argmax(q_pi)), debug


# ============================================================================
# WORLD SIMULATION
# ============================================================================

class World:
    """
    Simulation world containing multiple agents.
    
    Manages agent initialization, stepping, and history tracking.
    """
    
    def __init__(self, cfg: Dict, use_memory: bool = True):
        """
        Initialize the simulation world.
        
        Args:
            cfg: Configuration dictionary
            use_memory: Whether agents use episodic memory
        """
        self.cfg = cfg
        self.use_memory = use_memory
        self.num_agents = int(cfg["num_agents"])
        self.time_step = 0
        
        self._init_agents()
        
        # History tracking
        self.history = {
            "positions": [],
            "velocities": [],
            "crossings": [],
            "policies": [],
            "mean_episodes": [],
            "mean_y": [],
        }

    def _init_agents(self) -> None:
        """Initialize agents with random positions and velocities in safe zone."""
        self.agents: List[Agent] = []
        
        bx = float(self.cfg["world_bound_x"])
        by = float(self.cfg["world_bound_y"])
        
        for _ in range(self.num_agents):
            # Start in safe zone (negative y)
            pos_x = random.uniform(-bx * 0.8, bx * 0.8)
            pos_y = random.uniform(-by * 0.8, -by * 0.2)
            
            # Random initial velocity
            speed = random.uniform(0.5, self.cfg["target_speed"])
            angle = random.uniform(-math.pi, math.pi)
            vel_x = speed * math.cos(angle)
            vel_y = speed * math.sin(angle)
            
            self.agents.append(Agent(pos_x, pos_y, vel_x, vel_y, self.cfg))

    def step(self) -> None:
        """Advance simulation by one time step."""
        crossings = 0
        policies: List[int] = []
        num_episodes: List[int] = []

        for agent in self.agents:
            crossed, policy, dbg = agent.step(use_memory=self.use_memory)
            
            if crossed:
                crossings += 1
            policies.append(policy)
            num_episodes.append(dbg["num_episodes"])

        # Record history
        self.history["positions"].append([
            (a.pos_x, a.pos_y) for a in self.agents
        ])
        self.history["velocities"].append([
            (a.vel_x, a.vel_y) for a in self.agents
        ])
        self.history["crossings"].append(crossings)
        self.history["policies"].append(policies)
        self.history["mean_episodes"].append(
            float(np.mean(num_episodes)) if num_episodes else 0.0
        )
        self.history["mean_y"].append(
            float(np.mean([a.pos_y for a in self.agents]))
        )

        self.time_step += 1

    def run(self, num_steps: int, verbose: bool = True) -> None:
        """
        Run simulation for a specified number of steps.
        
        Args:
            num_steps: Number of steps to simulate
            verbose: Whether to print progress
        """
        for t in range(num_steps):
            self.step()
            
            if verbose and (t % 100 == 0):
                mode = "MEM" if self.use_memory else "NO_MEM"
                print(
                    f"[{mode}] t={t:4d}: "
                    f"crossings={sum(self.history['crossings']):4d}, "
                    f"mean_y={self.history['mean_y'][-1]:+6.2f}, "
                    f"mem={self.history['mean_episodes'][-1]:5.1f}"
                )


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_bird_marker(px: float, py: float, vx: float, vy: float, size: float = 0.4) -> Polygon:
    """
    Create a bird-shaped polygon marker oriented along velocity.
    
    Args:
        px, py: Position
        vx, vy: Velocity (determines orientation)
        size: Marker size
    
    Returns:
        Matplotlib Polygon patch
    """
    speed = math.sqrt(vx**2 + vy**2)
    angle = math.atan2(vy, vx) if speed > 0.01 else 0.0
    
    # Bird shape (pointing right)
    points = np.array([
        [size, 0],           # Nose
        [-size*0.5, size*0.4],   # Left wing
        [-size*0.3, 0],          # Tail notch
        [-size*0.5, -size*0.4],  # Right wing
    ])
    
    # Rotate to match velocity direction
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, s], [-s, c]])
    rotated = points @ rotation
    
    # Translate to position
    return Polygon(rotated + [px, py], closed=True)


def create_animation(world: World, filepath: str, fps: int = 20, skip: int = 4) -> None:
    """
    Create an MP4 animation of the simulation.
    
    Args:
        world: Simulated world with history
        filepath: Output file path
        fps: Frames per second
        skip: Frame skip (use every skip-th frame)
    """
    print(f"Creating {filepath}...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bx = float(world.cfg["world_bound_x"])
    by = float(world.cfg["world_bound_y"])
    nb = int(world.cfg["num_y_bands"])

    # Set up axes
    ax.set_xlim(-bx * 1.05, bx * 1.05)
    ax.set_ylim(-by * 1.15, by * 1.15)
    ax.set_aspect("equal")

    # Draw harm zone
    ax.add_patch(Rectangle((-bx, 0), 2*bx, by, alpha=0.2, color="red"))
    ax.axhline(y=0, color="red", linestyle="--", linewidth=2)

    # Draw y-band boundaries
    band_height = 2 * by / nb
    for i in range(1, nb):
        ax.axhline(y=-by + i * band_height, color="blue", linestyle=":", alpha=0.3)

    # Labels
    ax.text(bx * 0.85, by * 0.7, "HARM", fontsize=12, color="red", 
            fontweight="bold", alpha=0.7, ha="center")
    ax.text(bx * 0.85, -by * 0.7, "SAFE", fontsize=12, color="blue", 
            fontweight="bold", alpha=0.6, ha="center")

    mode = "WITH MEMORY" if world.use_memory else "NO MEMORY"
    ax.text(0, by * 1.05, mode, fontsize=14, ha="center", fontweight="bold",
            color="purple" if world.use_memory else "gray")

    # Animation setup
    frames = list(range(0, len(world.history["positions"]), skip))
    birds = []
    title = ax.set_title("")

    def update(frame_num):
        nonlocal birds
        
        # Remove old birds
        for b in birds:
            b.remove()
        birds = []

        # Get data for this frame
        t = frames[frame_num]
        positions = world.history["positions"][t]
        velocities = world.history["velocities"][t]

        # Draw agents
        for (px, py), (vx, vy) in zip(positions, velocities):
            color = "red" if py > 0 else "green"
            bird = create_bird_marker(px, py, vx, vy, 0.5)
            bird.set_facecolor(color)
            bird.set_edgecolor("black")
            bird.set_linewidth(0.5)
            bird.set_alpha(0.8)
            ax.add_patch(bird)
            birds.append(bird)

        # Update title
        total = sum(world.history["crossings"][:t+1])
        recent = sum(world.history["crossings"][max(0, t-30):t+1])
        mean_y = world.history["mean_y"][t]
        title.set_text(f"t={t}  crossings={total}  recent30={recent}  mean_y={mean_y:+.2f}")
        
        return birds + [title]

    # Create and save animation
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000//fps, blit=False)
    anim.save(filepath, writer=FFMpegWriter(fps=fps))
    plt.close()
    
    print(f"Saved {filepath}")


def plot_comparison(w_mem: World, w_no: World, filepath: Optional[str] = None) -> None:
    """
    Create comparison plots between memory and no-memory conditions.
    
    Args:
        w_mem: World simulated with memory
        w_no: World simulated without memory
        filepath: Output file path (if None, displays interactively)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cumulative crossings
    ax = axes[0, 0]
    ax.plot(np.cumsum(w_no.history["crossings"]), "r-", label="No Memory", lw=2)
    ax.plot(np.cumsum(w_mem.history["crossings"]), "b-", label="With Memory", lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Crossings")
    ax.set_title("Total Crossings into Harm Zone")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean Y position
    ax = axes[0, 1]
    ax.plot(w_no.history["mean_y"], "r-", label="No Memory", alpha=0.7)
    ax.plot(w_mem.history["mean_y"], "b-", label="With Memory", alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Y Position")
    ax.set_title("Mean Vertical Position (0 = boundary)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Memory size
    ax = axes[1, 0]
    ax.plot(w_mem.history["mean_episodes"], "b-", lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Episodes per Agent")
    ax.set_title("Memory Size Growth")
    ax.grid(True, alpha=0.3)

    # Plot 4: Early vs late comparison
    ax = axes[1, 1]
    T = len(w_mem.history["crossings"])
    T10 = max(1, T // 10)
    
    early_no = sum(w_no.history["crossings"][:T10])
    late_no = sum(w_no.history["crossings"][-T10:])
    early_mem = sum(w_mem.history["crossings"][:T10])
    late_mem = sum(w_mem.history["crossings"][-T10:])

    x = [0, 1]
    width = 0.35
    ax.bar([p - width/2 for p in x], [early_no, late_no], width, 
           label="No Memory", color="red", alpha=0.7)
    ax.bar([p + width/2 for p in x], [early_mem, late_mem], width, 
           label="With Memory", color="blue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Early (0-10%)", "Late (90-100%)"])
    ax.set_ylabel("Crossings")
    ax.set_title(f"Early vs Late Crossings\nNo: {early_no}→{late_no}  Mem: {early_mem}→{late_mem}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Affect-Based Episodic Memory: Algorithm A1-A8", fontsize=13)
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath, dpi=150)
        print(f"Saved {filepath}")
    
    plt.close()


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment(cfg: Dict, seeds: List[int], num_steps: int = 2000) -> Dict:
    """
    Run comparative experiment across multiple seeds.
    
    Args:
        cfg: Configuration dictionary
        seeds: List of random seeds for reproducibility
        num_steps: Steps per simulation
    
    Returns:
        Results dictionary with crossing statistics
    """
    results = {
        "no_mem": [], "mem": [],
        "no_early": [], "mem_early": [],
        "no_late": [], "mem_late": []
    }

    for i, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"Seed {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*50}")

        # Run without memory
        random.seed(seed)
        np.random.seed(seed)
        w_no = World(cfg, use_memory=False)
        w_no.run(num_steps, verbose=True)

        # Run with memory (same initial conditions)
        random.seed(seed)
        np.random.seed(seed)
        w_mem = World(cfg, use_memory=True)
        w_mem.run(num_steps, verbose=True)

        # Compute statistics
        T = len(w_mem.history["crossings"])
        T10 = max(1, T // 10)

        results["no_mem"].append(sum(w_no.history["crossings"]))
        results["mem"].append(sum(w_mem.history["crossings"]))
        results["no_early"].append(sum(w_no.history["crossings"][:T10]))
        results["mem_early"].append(sum(w_mem.history["crossings"][:T10]))
        results["no_late"].append(sum(w_no.history["crossings"][-T10:]))
        results["mem_late"].append(sum(w_mem.history["crossings"][-T10:]))

        print(f"No Mem: {results['no_mem'][-1]} "
              f"(early={results['no_early'][-1]}, late={results['no_late'][-1]})")
        print(f"Mem:    {results['mem'][-1]} "
              f"(early={results['mem_early'][-1]}, late={results['mem_late'][-1]})")

        # Generate visualizations for first seed only
        if i == 0:
            import os
            os.makedirs("outputs", exist_ok=True)
            plot_comparison(w_mem, w_no, "outputs/harm-comparison.png")
            create_animation(w_no, "outputs/harm-no_memory.mp4", skip=3)
            create_animation(w_mem, "outputs/harm-with_memory.mp4", skip=3)

    return results


def print_summary(results: Dict) -> None:
    """Print summary statistics from experiment results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Total crossings:")
    print(f"  No Memory: {np.mean(results['no_mem']):.1f} ± {np.std(results['no_mem']):.1f}")
    print(f"  Memory:    {np.mean(results['mem']):.1f} ± {np.std(results['mem']):.1f}")
    
    print(f"\nEarly phase (first 10%):")
    print(f"  No Memory: {np.mean(results['no_early']):.1f}")
    print(f"  Memory:    {np.mean(results['mem_early']):.1f}")
    
    print(f"\nLate phase (last 10%):")
    print(f"  No Memory: {np.mean(results['no_late']):.1f}")
    print(f"  Memory:    {np.mean(results['mem_late']):.1f}")
    
    if np.mean(results["no_mem"]) > 0:
        reduction = (
            (np.mean(results["no_mem"]) - np.mean(results["mem"])) 
            / np.mean(results["no_mem"]) * 100.0
        )
        print(f"\nOverall reduction: {reduction:+.1f}%")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AFFECT-BASED EPISODIC MEMORY DEMO")
    print("Algorithm A1-A8 (Separation Witness for Q1)")
    print("=" * 60)
    
    cfg = DEFAULT_CONFIG.copy()
    
    print("\nConfiguration:")
    print(f"  Needs: Safety (y>0 = harm), Motion (target speed)")
    print(f"  Policies: Flee (south), Thrust (forward)")
    print(f"  Memory: K={cfg['memory_retrieval_k']}, threshold={cfg['memory_store_threshold']}")
    print(f"  Reliability weighting: {cfg['use_reliability_weighting']}")
    print()

    seeds = [123, 456, 789, 1011]
    results = run_experiment(cfg, seeds, num_steps=2500)
    print_summary(results)
    
    print("\nDone!")
