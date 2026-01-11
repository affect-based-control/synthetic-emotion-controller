# Synthetic Emotion Controller

Implementation of an affect-based control architecture for autonomous agents, as described in:

> **Synthetic Emotions and Consciousness: Exploring Architectural Boundaries**

This repository demonstrates how emotion-like mechanisms can guide adaptive behavior, with examples showing both reactive affect-based control and episodic affect memory for anticipatory decision-making.

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

### Example 1: Social Flocking (`flock_no_memory.py`)

Demonstrates the basic emotion architecture **without episodic memory**:

- Agents navigate a 2D world with social needs (affiliation, safety, motion, coherence)
- Emotional signals (valence, arousal, drives) emerge from need satisfaction/frustration
- Policies include Seek, Avoid, Align, and Explore
- Agents exhibit **emergent flocking behavior** driven purely by affect

**Key insight for flocking:**
Heading coherence is treated as a *need*, not just a mechanical rule. When an agent's heading differs from neighbors, it experiences discomfort ("out of sync"), which drives the Align policy. This is biologically plausible—social animals experience discomfort when not coordinating with the group.

**Algorithm mapping:**
- A1 (Categorize): `categorize()` → density, crowding, coherence, motion
- A2 (Need Appraisal): `NeedSystem.assess()` and `compute_affect()`
- A3 (Episodic Retrieval): Not implemented (stateless demo)
- A4 (Policy Mix): `PolicySystem.get_policy_mix()`
- A5 (Action Scoring): `PolicySystem.score_actions()`
- A6-A7 (Execute): Position and heading updates

### Example 2: Anticipatory Harm Avoidance (`affect_memory_demo.py`)

Demonstrates **episodic affect memory** enabling anticipatory behavior:

- Agents explore a 2D world with a harm zone (y > 0) that delivers "electric shock"
- Shock causes strong negative affect, but only upon contact (no warning signal)
- **Without memory**: Agents rely on reactive responses, frequently entering harm zone
- **With memory**: Agents recall negative affect from similar past situations, triggering avoidance *before* crossing into harm

**Key mechanism:**
- Category (c) is **coarse** for memory retrieval (y-bands spanning the boundary)
- Affect (z) is computed from **raw** sensory state (actual y position)
- An agent at y = -1 (safe, no shock) retrieves memories from y = +1 (shocked, same band)
- This provides **anticipatory warning** that immediate perception alone cannot provide

**Algorithm structure:**
```
A1: Observe and categorize → (c, y) from x
A2: Assess needs → z_need, h_need  
A3: Retrieve episodic memory → z_mem, h_mem (K=1 retrieval)
A4: Fuse affect and hints → z, h, q(π)
A5: Instantiate policy → action scores
A6-A7: Execute and reappraise → z*, succ*
A8: Store episode (only if |succ*| > threshold)
```

**Memory depth and E3 compliance:**
The implementation uses K=1 retrieval, matching the formal Q1 argument in the paper. Despite this, agents typically saturate at memories containing only *two* salient episodes: one encoding negative affect upon entering harm, the other encoding positive affect upon escaping. This shallow memory depth provides strong evidence for E3 (minimal temporal binding) compliance: the architecture achieves genuine anticipatory behavior while storing far too few episodes for meaningful temporal integration or autobiographical binding. What persists in memory are primitive affect tags ("it felt bad there," "it felt good leaving"), not narrative structure.

**Expected results:**
- Early crossings: Similar for both conditions (learning period)
- Late crossings: ~0 with memory vs. continued crossings without memory
- Late crossing reduction: ~100%

## Running the Examples

### Example 1: Flocking
```bash
python flock_no_memory.py
```

Outputs (in `outputs/` directory):
- `flock_trajectories.png` - Agent paths colored by time
- `flock_snapshots.png` - Flock configuration at key timesteps
- `flock_metrics.png` - Alignment, cohesion, arousal, policy mix over time
- `flock_animation.mp4` - Animation of flock formation

### Example 2: Affect Memory Demo
```bash
python affect_memory_demo.py
```

Outputs (in `outputs/` directory):
- `harm-algorithm_comparison.png` - Comparison plots (crossings, memory size, etc.)
- `harm-no_memory.mp4` - Animation without memory
- `harm-with_memory.mp4` - Animation with memory

## Configuration

### Flocking Example

Key parameters in `cfg` dictionary:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_agents` | Number of flocking agents | 25 |
| `n_coherence_target` | Desired alignment level (higher = stronger flocking) | 0.85 |
| `drive_alpha` | Sensitivity of drives to need deviation | [2.0, 3.0, 2.5, 4.0] |
| `heading_smooth` | Turning inertia (lower = smoother) | 0.20 |
| `R_comfort` | Comfortable neighbor distance | 7.0 |
| `R_too_close` | Personal space radius | 2.0 |

### Affect Memory Example

Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_agents` | Number of agents | 30 |
| `n_y_bands` | Categorical bands for memory retrieval | 5 |
| `memory_K` | Number of episodes retrieved (k-NN) | 1 |
| `store_threshold` | Only store episodes with \|succ*\| > threshold | 0.4 |
| `alpha_mem` | Weight for memory-based hints | 0.85 |
| `alpha_z_mem` | Weight for memory affect fusion | 0.9 |
| `harm_intensity` | Strength of negative affect in harm zone | 0.95 |

## Design Notes

### Policy-to-Action Mappings

The present algorithm learns *when* to activate policies (anticipatory triggering) but not *how* to execute them. Policy-to-action mappings are assumed innate. For example:

- **Flocking**: The Align policy template scores headings by angular proximity to neighbor consensus
- **Harm avoidance**: The Flee policy directs agents toward a predetermined safe region

Learning effective policy-to-action mappings would require extended temporal windows for credit assignment, linking specific actions to delayed outcomes—left for future work.

### Episode Length and Temporal Credit Assignment

Episodes are kept to the minimum temporal size of a single step. This choice is intertwined with the innate policy-to-action assumption: with single-step episodes, an agent can learn *when* to activate a flee policy, but not *where* fleeing should lead. The latter would require persistence across multiple steps plus subsequent credit assignment.

### Selective Episode Storage

Episodes are only stored for highly successful or unsuccessful outcomes (|succ*| > threshold). This prevents memory from being swamped by neutral episodes that would dilute the signal from significant experiences.

### Heading Persistence

Both examples include optional heading persistence (smoothing/inertia) that prefers smooth movement trajectories. This is mechanical (like physical momentum), not memory-based, and does not constitute autobiographical temporal binding.

### k-NN Safety for Any K

While the implementation uses K=1 for simplicity and alignment with the formal Q1 argument, the k-NN formulation in Steps 6-8 of the algorithm remains safe for any K: weighted averaging of affect and hints does not create cross-episode summaries or violate the exclusion criteria E1-E4.

## Citation

If you use this repository, please cite:
```bibtex
@article{author2026synthetic,
  title={{Synthetic {E}motions and {C}onsciousness: {E}xploring {A}rchitectural {B}oundaries}},
  author={name},
  journal={AI \& Society},
  year={2026},
  publisher={Springer}
}
```

## License

MIT License

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
