# Synthetic Emotion Controller

Implementation of an affect-based control architecture for autonomous agents, as described in:

> **Synthetic Emotions and Consciousness: Exploring Architectural Boundaries**  
> Hermann Borotschnig — *Accepted for publication*   (2026) AI & Society, Springer Nature

This repository demonstrates how emotion-like mechanisms can guide adaptive behavior, with examples showing both the complete emotion-like control architecture (A1-A8) and graceful degradation under memory failure.

## Overview

The architecture implements a hierarchical control loop (A1-A8) where:

- **Upward abstraction (A1)**: Raw sensory input is categorized into abstract situations
- **Need appraisal (A2)**: Emotional signals arise from homeostatic need satisfaction/frustration
- **Episodic retrieval (A3)**: Memory provides affect and policy hints from similar past situations
- **Affective integration (A4)**: Hints from needs, affect, and memory are fused to select policies
- **Policy instantiation (A5)**: Abstract policies are instantiated into concrete actions
- **Execution and reappraisal (A6-A7)**: Actions are executed and outcomes evaluated
- **Episode storage (A8)**: Significant experiences are stored for future retrieval

## Example Implementations

### Example 1: Anticipatory Harm Avoidance (`affect_memory_demo.py`)

**This is the primary demonstration: a complete implementation of the A1-A8 emotion-like control architecture that respects constraints R1-R4, serving as a separation witness for research question Q1.**

- Agents explore a 2D world with a harm zone (y > 0) that delivers "electric shock"
- Shock causes strong negative affect, but only upon contact (no warning signal)
- **Without memory**: Agents rely on reactive responses, frequently entering harm zone
- **With memory**: Agents recall negative affect upon approach from similar past situations, triggering avoidance *before* crossing into harm

**Separation Witness Status:**
This implementation demonstrates that emotion-like control (A1-A8) can be realized while satisfying the risk-reduction constraints:
- **R1 (No shared workspace)**: Memory returns only aggregate hints and affect; no global broadcast
- **R2 (No metarepresentation)**: All stored quantities are first-order (categories, affect, hints); no self-descriptive tokens
- **R3 (No autobiographical consolidation)**: Episodes are atomic single-step records; no cross-episode summaries or narrative binding
- **R4 (Bounded learning)**: H matrices are fixed; credit assignment stays within module boundaries

**Key mechanism:**
- Category (c) is **coarse** for memory retrieval (y-bands spanning the boundary)
- Affect (z) is computed from **raw** sensory state (actual y position: shock only when "touching")
- An agent at y = -1 (safe, no shock) retrieves memories from y = +1 (shocked, same band)
- This provides **anticipatory warning** that immediate perception alone cannot provide

**Algorithm structure:**
```
A1: Observe and categorize → (c_t, y_t) from x_t
A2: Assess needs → z^need_t, h^need_t  
A3: Retrieve episodic memory → z^mem_t, h^mem_t, reliability
A4: Fuse affect and hints → z_t, h_t, q(π)
A5: Instantiate policy → action scores s_t(u)
A6-A7: Execute and reappraise → z*_t, succ*_t
A8: Store episode (only if |succ*| ≥ threshold)
```

**Normalization strategy:**
- **H matrices**: H_need and H_affect are row-L1 normalized at initialization, ensuring hints `h = H @ x` remain bounded in [-1, 1] when inputs are in [-1, 1]
- **Hint fusion**: Weights (α_need, α_mem, α_aff) are always normalized to sum to 1, keeping fused hints bounded
- **Memory retrieval**: Uses `h_mem = Σ w_j * succ_j * h_j` where `w_j = softmax(τ × sim_j)`. No denominator normalization—success magnitude scales hint strength linearly
- **Success computation**: Computed from need-based affect (z^need, z*^need), not fused affect, preventing "fear relief" artifacts

**Credit assignment:**
One-hot policy tagging isolates credit to the responsible policy. The "responsible policy" is identified as π_hat = argmax_π [q(π) * s̃_π(u_t)], and only that policy's one-hot tag is stored.

**Category design:**
```
c_t = [y_band one-hot, sign(v_y) one-hot, harm_flag one-hot]
```
This expanded representation reduces spurious cosine-similarity ties.


**Expected results:**
- Early crossings: Similar for both conditions (learning period)
- Late crossings: ~0 with memory vs. continued crossings without memory
- Late crossing reduction: ~80%

---

### Example 2: Social Flocking (`flock_no_memory.py`)

Demonstrates **graceful degradation** of the architecture when episodic memory is unavailable.

**Note:** This is a reduced architecture with A3, A7, A8 disabled. It does *not* constitute full emotion-like control as defined in the paper (which requires dual-source integration of needs and episodic memory). Rather, it tests whether the need-based pathway alone provides useful reactive control when memory systems fail or are deliberately disabled.

- Agents navigate a 2D world with social needs (affiliation, safety, motion, coherence)
- Emotional signals (valence, arousal, drives) emerge from need satisfaction/frustration
- Policies include Seek, Avoid, Align, and Cruise
- Agents exhibit **emergent flocking behavior** driven purely by reactive affect

**Key insight for flocking:**
Heading coherence is treated as a *need*, not just a mechanical rule. When an agent's heading differs from neighbors, it experiences discomfort ("out of sync"), which drives the Align policy. This is biologically plausible—social animals experience discomfort when not coordinating with the group.

**Conventions:**
- **Drives** follow the convention: `d_i = tanh(α_i × (n◇_i - n_i))`, where positive drive indicates deficit ("want more") and negative drive indicates surplus
- **Valence** follows standard psychological convention: `v_i = -d_i`, where positive valence means feeling good (need satisfied) and negative valence means feeling bad
- **Arousal** increases with absolute drive magnitude (distance from target in either direction)

**Simplifications in this baseline:**
- Speed is constant (heading is the only control variable)
- Action-selection temperature τ₂(a) ≡ const (not arousal-dependent)
- Episodic memory is disabled (A3, A7, A8 skipped)

**Affective attractor:**
When the flock reaches stable coordinated motion, the affect-space trajectory (valence vs. arousal) converges to a region on a one-dimensional curve. This "affective attractor" represents the characteristic emotional signature of stable flocking: mild positive valence, moderate arousal.

**Algorithm mapping:**
- A1 (Categorize): `categorize()` → density, crowding, coherence, motion
- A2 (Need Appraisal): `NeedSystem.assess()` and `compute_affect()`
- A3 (Episodic Retrieval): Skipped (graceful degradation test)
- A4 (Policy Mix): `PolicySystem.get_policy_mix()`
- A5 (Action Scoring): `PolicySystem.score_actions()`
- A6 (Execute): Position and heading updates
- A7 (Reappraisal): Skipped (graceful degradation test)
- A8 (Store episode): Skipped (graceful degradation test)

## Running the Examples

### Example 1: Affect Memory Demo (Separation Witness)
```bash
python affect_memory_demo.py
```

Outputs (in `outputs/` directory):
- `harm-comparison.png` - Comparison plots (crossings, memory size, mean Y position)
- `harm-no_memory.mp4` - Animation without memory
- `harm-with_memory.mp4` - Animation with memory

### Example 2: Flocking (Graceful Degradation)
```bash
python flock_no_memory.py
```

Outputs (in `outputs/` directory):
- `flock_trajectories.png` - Agent paths colored by time
- `flock_snapshots.png` - Flock configuration at key timesteps
- `flock_metrics.png` - Alignment, cohesion, valence, arousal, affect trajectory, policy mix
- `flock_animation.mp4` - Animation of flock formation

## Configuration

### Example 1: Affect Memory Demo

Key parameters in `DEFAULT_CONFIG` dictionary:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_agents` | Number of agents | 30 |
| `num_y_bands` | Categorical bands for memory retrieval | 6 |
| `memory_retrieval_k` | Number of episodes retrieved (k-NN) | 1 |
| `memory_store_threshold` | Only store episodes with \|succ*\| ≥ threshold | 0.25 |
| `alpha_hint_need` | Base weight for need-based hints | 0.30 |
| `alpha_hint_memory` | Base weight for memory-based hints | 0.50 |
| `alpha_hint_affect` | Base weight for affect-based hints | 0.20 |
| `beta_affect_fusion` | Base weight for need affect in z fusion | 0.50 |
| `use_reliability_weighting` | Scale memory weight by retrieval similarity | False |
| `success_mode` | Success measure: "drive", "emotion", or "hybrid" | "hybrid" |
| `success_omega` | Blend factor for hybrid success | 0.50 |
| `success_weight_safety` | Weight for safety channel in hedonic success | 4.0 |
| `success_weight_motion` | Weight for motion channel in hedonic success | 1.0 |

### Example 2: Flocking

Key parameters in `cfg` dictionary:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_agents` | Number of flocking agents | 25 |
| `speed` | Agent speed (constant, not controlled) | 0.85 |
| `n_affiliation_target` | Desired density level | 0.4 |
| `n_safety_target` | Desired personal space level | 0.9 |
| `n_motion_target` | Desired movement level | 0.9 |
| `n_coherence_target` | Desired alignment level (higher = stronger flocking) | 0.85 |
| `drive_alpha` | Sensitivity of drives to need deviation | [2.0, 3.0, 2.5, 4.0] |
| `heading_smooth` | Turning inertia (lower = smoother) | 0.20 |
| `tau_2` | Action selection temperature (constant) | 0.15 |
| `R_comfort` | Comfortable neighbor distance | 7.0 |
| `R_too_close` | Personal space radius | 2.0 |

## Design Notes

### Separation Witness: Full A1-A8 Implementation

The `affect_memory_demo.py` implements the complete emotion-like control loop as specified in the paper:
- **Dual-source integration**: Both need-based appraisal (A2) and episodic memory (A3) contribute to policy selection
- **Bounded interfaces**: Memory returns only aggregate quantities; no raw episodes cross module boundaries
- **First-order representations**: All stored quantities describe situations and outcomes, not self-states
- **Single-step atomicity**: Episodes encode one action-outcome pair; no temporal binding across steps

### Graceful Degradation: Reduced Architecture

The `flock_no_memory.py` demonstrates that the need-based pathway alone provides useful reactive control when episodic memory is unavailable. This robustness property may be valuable in deployments where memory systems could fail.

### Reliability Weighting: Design Choice

The `use_reliability_weighting` flag controls whether retrieval similarity affects fusion weights:

- **False (default)**: Similarity is used ONLY in retrieval via `w_j = softmax(τ × sim_j)`. Fusion uses fixed weights. 

- **True**: Similarity affects BOTH retrieval weights AND fusion weights (α_mem scaled by reliability before normalization). This may over-attenuate memory contribution.

The default setting (False) lets the retrieval weights `w_j` handle similarity, and uses fixed fusion weights (when no good matches exist, `w_j` becomes diffuse and `h_mem` becomes diffuse).

### Policy-to-Action Mappings

The present algorithm learns *when* to activate policies (anticipatory triggering) but not *how* to execute them. Policy-to-action mappings are assumed innate:
- **Harm avoidance**: Flee policy accelerates south (innate safe direction); Thrust policy maintains current velocity
- **Flocking**: Align policy scores headings by angular proximity to neighbor consensus

Learning effective policy-to-action mappings would require extended temporal windows for credit assignment—left for future work.

### Episode Storage Strategy

Episodes are only stored when outcomes are significant (|succ*| ≥ threshold). This prevents memory dilution from neutral experiences while ensuring that both positive and negative outcomes contribute to learning.

### k-NN Safety for Any K

While the implementation uses K=1, the k-NN formulation would remain R1-R4 compliant for any K: weighted averaging of affect and hints from similar episodes does not create cross-episode summaries or autobiographical binding.

## Citation

If you use this repository, please cite:
```bibtex
@article{borotschnig2026synthetic,
  title={{Synthetic Emotions and Consciousness: Exploring Architectural Boundaries}},
  author={Borotschnig, Hermann},
  journal={AI \& Society},
  year={2026},
  publisher={Springer Nature}
}
```

## License

MIT License

Copyright (c) 2026 Hermann Borotschnig

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
