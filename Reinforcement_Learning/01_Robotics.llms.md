# Robotics

Getting a policy to move a physical machine: why imitation beat reinforcement learning in practice, what action chunking fixed, how vision-language-action models arrived in 2024-2026, and runnable code that trains a behaviour-cloning policy on real LeRobot arm data and grounds a spoken instruction into a 3D pick target from the webcam.

Author

Benedict Thekkel

## 1. What is Robot Learning?

Robot learning is the task of producing a **policy** $\pi(a_t \mid o_t)$ that maps what a robot senses to what it should do next, learned from data rather than derived from a hand-written controller. What separates it from every other task in this repo is that **the output is executed** - the model’s mistakes move a physical machine, and there is no way to undo them.

**Input (observation).** Usually several streams at once:

- **Proprioception** - joint positions, velocities, torques. Low-dimensional, exact, always available. The SO-101 arm used below reports 6 numbers at 30 Hz.
- **Vision** - one or more cameras, often a wrist camera plus a scene camera. High-dimensional, and the only thing that tells the robot where the object is.
- **Force/torque and tactile** - contact, slip, grasp quality. Sparse in datasets and critical in contact-rich tasks.
- **Language** - the instruction. This is what turned single-task policies into general ones.

**Output (action).** The action space choice determines almost everything about the system:

| Action space | What it is | Trade-off |
|----|----|----|
| Joint torques | Direct motor commands | Maximum control authority; needs kHz rates and is dangerous to learn from scratch |
| Joint positions | Target angles for a low-level PD controller | **The standard for learned policies**; the controller absorbs the dynamics |
| End-effector pose (delta or absolute) | Where the gripper should be | Transfers across robots with different arms; needs inverse kinematics |
| Discretised action tokens | Actions as vocabulary entries | Lets a language model emit actions directly (RT-2, OpenVLA) |
| Action chunk | The next $K$ actions in one prediction | Fixes compounding error and jerk (section 10) - now near-universal |

**What makes it hard, specifically:**

- **Compounding error.** A policy trained on expert data sees only expert states. One small mistake puts it in a state the training data never covered, where its next action is worse - and the error compounds quadratically in the episode length (Ross and Bagnell, 2010).
- **Data is expensive and embodiment-specific.** Every trajectory costs a human teleoperating a real robot in real time. A dataset that would be “small” in vision is a month of work here, and it may not transfer to a robot with different joint limits.
- **The sim-to-real gap.** Simulation is free and wrong. Friction, contact, deformable objects and sensor noise are all approximations, and a policy can exploit the approximation.
- **Multimodality.** There are many correct ways to reach a mug. Averaging them with a mean-squared-error loss produces the one action that is between them, which is often the one that hits the mug. Section 9 measures this directly.
- **Safety is a hard constraint, not a metric.** You cannot explore by trying random torques on a 30 kg arm near a person.

**Neighbouring tasks:**

| Task | What it does | Typical tool |
|----|----|----|
| Reinforcement learning | Learn from reward, usually in simulation | see `00_Reinforcement_Learning` |
| Imitation learning / BC | Learn from demonstrations, supervised | ACT, Diffusion Policy (sections 8-10) |
| Motion planning | Find a collision-free path with a known model | OMPL, RRT\*, cuRobo |
| Grasp synthesis | Where to place the fingers | GraspNet, Contact-GraspNet, AnyGrasp |
| Pose estimation | 6-DoF object pose from an image | FoundationPose, MegaPose |
| SLAM / navigation | Where am I, and how do I get there | ORB-SLAM3, Nav2 |
| Depth and detection | The perception front-end | see `Computer_Vision/00_Depth_Estimation`, `13_Zero_Shot_Object_Detection` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Warehouse picking | Logistics (Amazon Robotics, Covariant, Ocado) | Bin image + item request -\> grasp pose and motion | Enormous item diversity; picks per hour is the metric; a failed grasp is cheap, a dropped fragile item is not |
| Manufacturing assembly | Automotive, electronics | Part pose + force feedback -\> insertion trajectory | Sub-millimetre tolerance; contact-rich; cycle time measured in seconds |
| Mobile manipulation in homes and labs | Research and early products (1X, Figure, Physical Intelligence) | Language instruction + RGB -\> whole-body actions | Open-ended tasks; unstructured scenes; safety around people |
| Surgical robotics | Healthcare (da Vinci, autonomous suturing research) | Endoscope video + tool state -\> tool motion | Regulatory approval; near-zero error tolerance; almost always teleoperated, learning assists |
| Agricultural robotics | Agriculture | Crop image -\> harvest or spray action | Deformable, occluded, highly variable targets; outdoor lighting; cost per hectare |
| Autonomous driving | Automotive | Multi-sensor scene -\> trajectory | Safety certification; long tail of rare events; offline learning only |
| Drone inspection | Energy, infrastructure | Camera + IMU -\> flight path | Real-time onboard compute; wind; loss of the vehicle is total |
| Lab automation | Pharma, materials | Protocol + vision -\> pipetting, handling | Reliability over speed; provenance and reproducibility |
| Teleoperation assistance | Remote operations, prosthetics | Operator intent + sensors -\> corrected action | Shared autonomy; latency; the human is in the loop by design |

**What the success rate hides.**

- **“90% success” is measured on a specific table, in specific lighting, with specific objects.** Move the table, change the lighting, or swap in an unseen mug and the number moves a lot. Robot evaluation has very poor external validity, and the field knows it - which is why the strongest recent results are reported across many scenes and embodiments.
- **The evaluation itself is expensive and noisy.** 50 real-robot trials is a full day’s work, and the 95% confidence interval on 45/50 is roughly \[79%, 97%\]. Most reported differences between methods are inside the error bar of the number of trials actually run.
- **The demo is not the deployment.** A policy that works when a researcher resets the scene between attempts is a long way from one that runs for eight hours unattended, recovers from its own failures, and never damages anything.
- **Latency is a control problem, not an inference problem.** A 7B VLA at 5 Hz cannot close a feedback loop that needs 30 Hz. Action chunking exists partly to bridge exactly this - predict a chunk slowly, execute it quickly.
- **Sim results transfer unevenly.** Locomotion and free-space motion transfer well from simulation; contact-rich manipulation and deformables transfer badly. A method’s sim benchmark says little about its behaviour on a real gripper.

------------------------------------------------------------------------

## 3. How Modern Robot Learning Works

**1. Classical control and planning (1970s-2010s, still most of what runs).** Model the kinematics and dynamics, plan a collision-free path (RRT, PRM), track it with a PD or impedance controller. Reliable, verifiable, and the reason industrial robots work - but it needs an accurate model of the world, which is exactly what a cluttered home does not provide.

**2. Deep RL in simulation, then sim-to-real (2016-2021).** Train with massive parallel simulation and randomise the physics (friction, mass, latency, textures) so the policy cannot depend on any particular value - **domain randomisation** (Tobin et al., 2017). OpenAI’s Rubik’s-cube hand (2019) is the landmark. It works spectacularly for **locomotion**, where ANYmal and later quadrupeds learned robust walking entirely in sim, and much less well for contact-rich manipulation.

**3. Imitation learning takes over (2021 -\> now).** Collect human teleoperation demonstrations and train supervised. It is far more sample-efficient than RL on real hardware, needs no reward function, and cannot explore dangerously. Its weakness is the compounding-error problem, and the fixes were the big advances of 2023:

- **ACT** (Action Chunking with Transformers, Zhao et al., 2023) - predict a **chunk** of the next $K$ actions rather than one, and execute them with temporal ensembling. This removes most of the compounding error, cuts jerk, and made cheap bimanual hardware (ALOHA) work on fine tasks.
- **Diffusion Policy** (Chi et al., 2023) - model the action distribution with a denoising diffusion model instead of regressing the mean. This is the direct fix for **multimodality**: where MSE regression averages incompatible correct actions, a diffusion model samples one of them.

**4. Vision-language-action models (2022 -\> now), the current frontier.** Take a pretrained vision-language model and teach it to emit actions, so the robot inherits the VLM’s world knowledge and language grounding:

- **RT-1** (2022) - a transformer over image and language tokens, trained on 130k real episodes.
- **RT-2** (2023) - actions as **text tokens** in a VLM’s vocabulary, so web-scale pretraining transfers to control. This is where “pick up the extinct animal” started working.
- **Open X-Embodiment / RT-X** (2023) - 22 embodiments, 1M+ episodes pooled into one dataset; training across robots improved performance on each of them.
- **OpenVLA** (2024) - a 7B open VLA on Llama 2 + fused visual encoders, trained on 970k OXE episodes.
- **pi0 / pi0.5** (Physical Intelligence, 2024-2025) - a flow-matching action expert on top of a VLM, generating continuous 50 Hz action chunks; pi0.5 targets open-world generalisation to homes it has never seen.
- **GR00T N1 / N1.5** (NVIDIA, 2025) - an open humanoid foundation model with an explicit slow-VLM / fast-diffusion-action two-system split.
- **SmolVLA** (Hugging Face, 2025) - 450M parameters, trained on community LeRobot datasets, and designed to run on a consumer GPU or CPU. The interesting claim is not peak performance but that a usable VLA now fits on this notebook’s hardware.

**5. Where the field is in 2026.** The consensus recipe is: a pretrained VLM backbone, an action expert producing chunks (diffusion or flow matching), trained on pooled cross-embodiment data, fine-tuned on a few hundred task-specific demonstrations. The open problems are unchanged and hard: **data** (still the bottleneck; teleoperation does not scale like web scraping), **evaluation** (no agreed real-robot benchmark, and everyone’s table is different), **generalisation** to genuinely new objects and scenes, and **reliability** at the 99.9% level that deployment needs rather than the 90% that demos show.

------------------------------------------------------------------------

## 4. Evaluation Metrics

Robotics evaluation is the least standardised area in this repo, and the reason is structural: running the benchmark costs a physical robot and a human hour per few trials.

**Success rate.** The primary metric: the fraction of trials where the task was completed. What matters is what usually goes unreported - **how many trials**, **what counts as success**, and **what was randomised between them** (object pose only? object identity? lighting? the whole scene?).

**Its confidence interval, which is wide.** With $n$ trials and $k$ successes, use a Wilson or Clopper-Pearson interval, never $\pm$ a standard deviation. At $n = 20$, the 95% interval on 90% success stretches roughly from 70% to 97% - so a policy reported at 90% and one at 80% on 20 trials each are statistically indistinguishable. The cell below computes this.

**Partial-credit and progress metrics.** Binary success wastes information on long tasks, so many papers report sub-goal completion (approached / grasped / lifted / placed) or normalised progress.

**Action-prediction error (MSE / MAE on held-out demonstrations).** Cheap, offline, and **a weak proxy**: it measures agreement with the demonstrator, not task success. A policy can have low action error and fail (small errors at the critical contact moment), or high action error and succeed (a different but valid strategy). It is what section 11 measures, because it is what can be measured without a robot - and that limitation is the point.

**Throughput and latency.** Picks per hour in a warehouse; control frequency for the policy. A policy that needs 200 ms per inference cannot run a 30 Hz loop without action chunking.

**Safety and reliability.** Collisions, force limit violations, emergency stops, mean time between failures. These are the numbers a deployment cares about and papers rarely report.

**Simulation benchmarks** (Meta-World, RLBench, robosuite, LIBERO, CALVIN, ManiSkill) give reproducible numbers and standard splits. They are the right place for method comparison and the wrong place for claims about real robots.

------------------------------------------------------------------------

``` python
import numpy as np
from scipy import stats


def wilson_interval(successes, trials, confidence=0.95):
    "Wilson score interval - the right binomial CI for small n (not mean +/- sd)."
    if trials == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / trials
    denom = 1 + z ** 2 / trials
    centre = (p + z ** 2 / (2 * trials)) / denom
    half = z * np.sqrt(p * (1 - p) / trials + z ** 2 / (4 * trials ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


print("what a reported robot success rate actually pins down:\n")
for trials in (10, 20, 50, 200):
    k = int(round(0.90 * trials))
    lo, hi = wilson_interval(k, trials)
    print(f"  {k:3d}/{trials:3d} = 90.0%   95% CI [{lo:.1%}, {hi:.1%}]   width {hi - lo:.1%}")

# Two methods, the usual number of trials, the usual conclusion.
a_k, b_k, n = 18, 16, 20
lo_a, hi_a = wilson_interval(a_k, n)
lo_b, hi_b = wilson_interval(b_k, n)
_, p_value = stats.fisher_exact([[a_k, n - a_k], [b_k, n - b_k]])
print(f"\nmethod A {a_k}/{n} = {a_k / n:.0%}  CI [{lo_a:.0%}, {hi_a:.0%}]")
print(f"method B {b_k}/{n} = {b_k / n:.0%}  CI [{lo_b:.0%}, {hi_b:.0%}]")
print(f"Fisher exact p = {p_value:.2f} - a 10-point gap on 20 trials is not evidence of anything.")
```

    what a reported robot success rate actually pins down:

        9/ 10 = 90.0%   95% CI [59.6%, 98.2%]   width 38.6%
       18/ 20 = 90.0%   95% CI [69.9%, 97.2%]   width 27.3%
       45/ 50 = 90.0%   95% CI [78.6%, 95.7%]   width 17.0%
      180/200 = 90.0%   95% CI [85.1%, 93.4%]   width 8.4%

    method A 18/20 = 90%  CI [70%, 97%]
    method B 16/20 = 80%  CI [58%, 92%]
    Fisher exact p = 0.66 - a 10-point gap on 20 trials is not evidence of anything.

## 5. Datasets and Simulators

Robot data is the field’s bottleneck, so the important artefacts of the last three years are **datasets**, not architectures.

| Dataset / simulator | Contents | Scale | Type | License | Typical use |
|----|----|----|----|----|----|
| [Open X-Embodiment](https://robotics-transformer-x.github.io/) | Pooled trajectories from 21 institutions, 22 robot types | 1M+ episodes, 500+ skills | Real | CC BY 4.0 | **The pretraining corpus for every open VLA** |
| [DROID](https://droid-dataset.github.io/) | Franka arm, in-the-wild scenes, 3 cameras | 76k trajectories, 564 scenes | Real | MIT | Diverse single-arm manipulation |
| [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) | WidowX, kitchen-scale tasks with language | 60k trajectories | Real | CC BY 4.0 | Language-conditioned manipulation |
| [RoboMimic](https://robomimic.github.io/) | Proficient and multi-human demonstrations | ~10k trajectories | Sim + real | MIT | The standard imitation-learning ablation suite |
| [LeRobot datasets](https://huggingface.co/datasets?other=LeRobot) | Community datasets in one standard format on the Hub | 1000s of datasets | Real + sim | Apache 2.0 | **The practical entry point** |
| [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) | 2D block-pushing, the Diffusion Policy benchmark | 206 episodes, 25,650 frames | Sim | Apache 2.0 | **Used below** (section 9) - the multimodality demo |
| [lerobot/svla_so101_pickplace](https://huggingface.co/datasets/lerobot/svla_so101_pickplace) | Real SO-101 arm, 6 joints, pick and place, 2 cameras | 50 episodes, 11,939 frames | **Real** | Apache 2.0 | **Used below** (sections 8, 10-11) |
| [Meta-World](https://meta-world.github.io/) / [RLBench](https://sites.google.com/view/rlbench) | 50 / 100 scripted manipulation tasks | Unlimited | Sim | MIT / Apache 2.0 | Multi-task and meta-learning benchmarks |
| [LIBERO](https://libero-project.github.io/) / [CALVIN](http://calvin.cs.uni-freiburg.de/) | Lifelong / long-horizon language-conditioned suites | 130 / 34 tasks | Sim | MIT | **The current standard for comparing VLAs** |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) / [MuJoCo](https://mujoco.org/) / [ManiSkill](https://maniskill.readthedocs.io/) | GPU-parallel physics simulators | 1000s of parallel envs | Sim | BSD / Apache 2.0 | Where RL for robotics actually happens |

**This notebook uses two real LeRobot datasets**, both tiny by design. `svla_so101_pickplace` is **50 episodes of an actual SO-101 arm** doing pick-and-place, recorded at 30 Hz - 11,939 frames of 6-DoF joint positions, and the parquet is 0.4 MB because the video streams are separate files this notebook never downloads. `pusht` is the simulated 2D pushing task that Diffusion Policy was introduced on, and it is here for one specific lesson in section 9. Both are Apache 2.0 and neither is gated. Everything lands in `DL_tasks/datasets/`, which is gitignored.

**No simulator is installed.** Rolling out a policy needs MuJoCo or Isaac, which is a large dependency for a demonstration; instead the policies below are evaluated **open-loop on held-out episodes**, and section 4 is explicit about what that does and does not tell you.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

| Model | Year | Params | Backbone | Action output | Control rate | License | Best for |
|----|----|----|----|----|----|----|----|
| Behaviour cloning MLP | \- | ~0.1-1M | none | 1 action | any | \- | **The baseline** - built in section 8 |
| **ACT** | 2023 | ~80M | ResNet + transformer | **chunk of K** | ~50 Hz | MIT | Fine bimanual tasks on cheap hardware |
| **Diffusion Policy** | 2023 | ~70-250M | ResNet/ViT + U-Net or transformer | chunk, **sampled** | ~10 Hz | MIT | **Multimodal action distributions** |
| RT-1 | 2022 | 35M | EfficientNet + transformer | discretised tokens | 3 Hz | Apache 2.0 | The first large real-robot transformer |
| RT-2 | 2023 | 12-55B | PaLI-X / PaLM-E | action-as-text tokens | 1-5 Hz | closed | Web-knowledge transfer to control |
| **OpenVLA** | 2024 | 7B | Llama 2 + DINOv2/SigLIP | discretised tokens | ~5 Hz | MIT | **The open VLA reference**; needs ~16 GB VRAM |
| Octo | 2024 | 27-93M | transformer | diffusion head | ~10 Hz | MIT | Small, flexible, generalist |
| **pi0 / pi0.5** | 2024-25 | ~3B | PaliGemma + flow-matching expert | continuous chunk | **50 Hz** | Apache 2.0 (pi0 weights) | Dexterous, open-world tasks |
| **GR00T N1.5** | 2025 | ~2-3B | Eagle VLM + diffusion transformer | chunk | ~30 Hz | Apache 2.0 | Humanoids; explicit slow/fast split |
| **SmolVLA** | 2025 | **450M** | SmolVLM2 + flow-matching expert | chunk | ~30 Hz+ | Apache 2.0 | **Runs on this box**; community LeRobot data |
| RDT-1B | 2024 | 1B | diffusion transformer | chunk | ~10 Hz | MIT | Bimanual manipulation |

Benchmarks to watch: **[LIBERO](https://libero-project.github.io/)** and **CALVIN** for language-conditioned simulation results, and the per-paper real-robot tables that unfortunately remain the only way VLAs are compared on hardware.

**What wins what.** On dexterity and open-world generalisation, pi0.5 and GR00T lead. On accuracy-per-parameter and practical accessibility, SmolVLA is the notable 2025 result - 450M parameters is small enough to fine-tune on one consumer GPU, which changes who can participate. On raw control quality for a single well-defined task with a few hundred demonstrations, **ACT and Diffusion Policy are still the right answer** and a VLA is overkill.

**What runs on this box.** The 12 GB card fits SmolVLA (450M) and Octo comfortably, pi0 (~3B) in bf16 with care, and **not** OpenVLA-7B without quantisation. All of them require the `lerobot` package, which is a vendor runtime rather than a general-purpose library, so this notebook does not import it (see the general-purpose-library rule in `CLAUDE.md`); section 13 gives the command to install it. What *is* runnable here through plain `transformers` is the **perception front-end** that every one of these models contains - and that is what section 12 builds.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `torch` - the behaviour-cloning policies (written from scratch)
- `transformers` (\>=5.13) - OWLv2 (zero-shot detection) and Depth Anything V2, both transformers-native, for the perception demo
- `pandas` + `huggingface_hub` - reading the LeRobot parquet files directly
- `numpy` / `scipy` - kinematics, statistics, nearest-neighbour probes
- `opencv-python-headless` - webcam capture in section 12
- `pyecharts` - all charts (repo rule)

Memory: the BC policies are ~0.5M parameters, the two LeRobot parquets are under 2 MB combined, and the perception models are 1.24 GB (OWLv2) plus 0.10 GB (Depth Anything V2-Small). The whole notebook peaks around 2 GB of VRAM, well inside the 12 GB card - and the models are still freed between sections, per the house rule.

------------------------------------------------------------------------

``` python
# Everything runs through torch and transformers - no vendor robotics packages.
# %pip install -q torch transformers pandas huggingface_hub scipy opencv-python-headless pyecharts

# To actually run a VLA or roll out a policy on hardware or in sim:
# %pip install -q lerobot "gymnasium[mujoco]"
```

``` python
import ctypes
import ctypes.util
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv

# Knowledge/.env sets HF_TOKEN - authenticated Hub requests get higher rate limits
load_dotenv(find_dotenv(usecwd=True))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device != "cpu":
    print(torch.cuda.get_device_name(0))
print("device:", device)


def vram(tag=""):
    "Report current GPU memory (allocated / reserved). No-op on CPU."
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM {tag:20s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")


def free_memory(*objs):
    "Delete objects, run GC, empty the CUDA cache, and return freed RAM to the OS.\n\n    Note the argument list only drops *this function's* references. A name bound in\n    the notebook still holds the object, so the working idiom is `del model;\n    free_memory()` at the call site - which is what every section below uses.\n    "
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # glibc keeps freed CPU allocations in its arenas, so RSS ratchets upward across
    # sections; malloc_trim(0) hands them back. Not optional on a 20 GB box - see
    # dl-visualization-and-memory.md.
    try:
        ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim(0)
    except Exception:
        pass


def memory_report(tag=""):
    "Print current system RAM (and VRAM if a GPU is present), in GB."
    vm = psutil.virtual_memory()
    print(f"RAM  {tag:20s} {(vm.total - vm.available) / 1e9:5.2f} / {vm.total / 1e9:5.2f} GB")
    vram(tag)


DATA_DIR = Path("../../datasets")        # gitignored
DATA_DIR.mkdir(exist_ok=True)
HF_CACHE = str(DATA_DIR / "hf_cache")

torch.manual_seed(0)
memory_report("baseline")
```

    NVIDIA GeForce RTX 3060
    device: cuda:0
    RAM  baseline             15.03 / 20.97 GB
    VRAM baseline              0.00 GB allocated /  0.00 GB reserved

``` python
import json

from huggingface_hub import hf_hub_download


def load_lerobot(repo):
    "Read a LeRobot dataset's parquet + metadata directly - no `lerobot` package needed.\n\n    A LeRobot dataset is parquet for the low-dimensional streams plus separate MP4s for\n    the cameras. Everything here is joint positions, so only the (sub-megabyte) parquet\n    is fetched and the video files are never touched.\n    "
    table = hf_hub_download(repo, "data/chunk-000/file-000.parquet",
                            repo_type="dataset", cache_dir=HF_CACHE)
    meta = hf_hub_download(repo, "meta/info.json", repo_type="dataset", cache_dir=HF_CACHE)
    return pd.read_parquet(table), json.load(open(meta))


so101, so101_info = load_lerobot("lerobot/svla_so101_pickplace")

JOINTS = so101_info["features"]["observation.state"]["names"]
STATE = np.stack(so101["observation.state"].to_numpy()).astype("float32")
ACTION = np.stack(so101["action"].to_numpy()).astype("float32")
EPISODE = so101["episode_index"].to_numpy()
FPS = so101_info["fps"]

print(f"robot {so101_info['robot_type']}   {so101_info['total_episodes']} episodes   "
      f"{so101_info['total_frames']:,} frames at {FPS} Hz")
print(f"episode length: mean {np.bincount(EPISODE).mean():.0f} frames "
      f"({np.bincount(EPISODE).mean() / FPS:.1f} s)")
print(f"joints: {JOINTS}")
print(f"\ntask: {pd.read_parquet(hf_hub_download('lerobot/svla_so101_pickplace', 'meta/tasks.parquet', repo_type='dataset', cache_dir=HF_CACHE)).index.tolist()}")

# The action is the LEADER arm's position and the state is the FOLLOWER's - teleoperation
# through a position-controlled twin. So a_t is close to s_t, and the gap is the tracking
# lag. That gap is exactly what a one-step policy has to predict, and why it is nearly
# trivial: section 10 is about predicting further ahead than that.
print(f"\nmean |action - state| per joint (the tracking lag): "
      f"{np.abs(ACTION - STATE).mean(0).round(2)}")
```

    robot so100_follower   50 episodes   11,939 frames at 30 Hz
    episode length: mean 239 frames (8.0 s)
    joints: ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']

    task: ['pink lego brick into the transparent box']

    mean |action - state| per joint (the tracking lag): [4.14 3.83 3.63 2.21 1.66 2.34]

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Line

# One episode, all six joints. Read the shape of the task off it: the arm reaches out,
# the gripper closes, it lifts and traverses, the gripper opens.
EP = 0
sel = EPISODE == EP
t = np.arange(sel.sum()) / FPS

traj = Line().add_xaxis([round(float(v), 2) for v in t])
for j, name in enumerate(JOINTS):
    traj.add_yaxis(name, [round(float(v), 2) for v in STATE[sel][:, j]],
                   is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
traj.set_global_opts(
    title_opts=opts.TitleOpts(title=f"SO-101 joint trajectories, episode {EP}",
                              subtitle="a real pick-and-place recorded by teleoperation at 30 Hz"),
    xaxis_opts=opts.AxisOpts(type_="value", name="seconds"),
    yaxis_opts=opts.AxisOpts(name="joint position (degrees)"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
    datazoom_opts=[opts.DataZoomOpts(range_start=0, range_end=100)],
)
traj.render_notebook()
```

## 8. Behaviour Cloning: the baseline, and the two baselines below it

Behaviour cloning is supervised learning on demonstrations: given the observation, regress the action the human produced. It is the simplest thing that works, it needs no reward function and no simulator, and it is the foundation every method in section 6 builds on.

The policy below takes a short **history** of joint states (3 frames, so it can see velocity) and predicts the next action. Two baselines make the number meaningful:

- **Hold still** - predict $a_t = s_t$, “stay where you are”. On a position-controlled arm this is a real baseline rather than a straw man: the arm barely moves in 33 ms, so the only thing to predict is the teleoperation tracking lag, which is about 3 degrees. Any policy that does not clearly beat it has learned nothing.
- **Nearest neighbour** - find the most similar state in the training set and copy the action that followed. Non-parametric, no training, and it is the honest test of whether the network generalised or memorised.

The split is **by episode**, not by frame. Splitting randomly across frames would put frame $t$ in train and frame $t+1$ in test, which are nearly identical - the robotics version of the leakage warning in the tabular notebooks, and an easy way to report a policy that does not exist.

------------------------------------------------------------------------

``` python
HISTORY = 3            # frames of state the policy sees
TRAIN_EPISODES = np.arange(40)      # 40 train / 10 test, split BY EPISODE

state_mean, state_std = STATE.mean(0), STATE.std(0) + 1e-6


def build_windows(chunk_size, history=HISTORY):
    "Sliding windows: `history` past states -> the next `chunk_size` actions."
    xs, ys, eps = [], [], []
    for e in np.unique(EPISODE):
        idx = np.where(EPISODE == e)[0]
        for t in range(history - 1, len(idx) - chunk_size):
            xs.append(STATE[idx[t - history + 1: t + 1]].ravel())
            ys.append(ACTION[idx[t: t + chunk_size]])
            eps.append(e)
    return np.array(xs, dtype="float32"), np.array(ys, dtype="float32"), np.array(eps)


class BCPolicy(nn.Module):
    "An MLP from a state history to a chunk of future actions. The whole of BC."

    def __init__(self, in_dim, n_joints, chunk_size, hidden=512):
        super().__init__()
        self.chunk_size, self.n_joints = chunk_size, n_joints
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, chunk_size * n_joints))

    def forward(self, x):
        return self.net(x).view(-1, self.chunk_size, self.n_joints)


def train_bc(chunk_size, steps=3000, lr=1e-3, batch=256, seed=0):
    "Train a chunked BC policy; return it with its held-out predictions and targets."
    torch.manual_seed(seed)
    X, Y, eps = build_windows(chunk_size)
    train = np.isin(eps, TRAIN_EPISODES)

    x_mu, x_sd = np.tile(state_mean, HISTORY), np.tile(state_std, HISTORY)
    Xn = (X - x_mu) / x_sd
    Yn = (Y - state_mean) / state_std             # actions live in the same joint space

    policy = BCPolicy(X.shape[1], len(JOINTS), chunk_size).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.tensor(Xn[train], device=device)
    yt = torch.tensor(Yn[train], device=device)

    t0 = time.perf_counter()
    for step in range(steps):
        i = torch.randint(0, len(xt), (batch,), device=device)
        # L1, not MSE: action targets have outliers (fast teleoperator corrections) and
        # squared error lets them dominate. ACT uses L1 for the same reason.
        loss = F.l1_loss(policy(xt[i]), yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    seconds = time.perf_counter() - t0

    policy.eval()
    with torch.inference_mode():
        pred = policy(torch.tensor(Xn[~train], device=device)).cpu().numpy()
    pred = pred * state_std + state_mean
    return policy, dict(pred=pred, true=Y[~train], last_state=X[~train][:, -len(JOINTS):],
                        train_X=Xn[train], train_Y=Y[train], test_Xn=Xn[~train],
                        seconds=seconds, params=sum(p.numel() for p in policy.parameters()))


policy1, run1 = train_bc(chunk_size=1)
print(f"BC (1 action)   {run1['params']:,} params, trained in {run1['seconds']:.0f}s on "
      f"{len(run1['train_X']):,} windows")

mae_bc = float(np.abs(run1["pred"][:, 0] - run1["true"][:, 0]).mean())
mae_hold = float(np.abs(run1["last_state"] - run1["true"][:, 0]).mean())
print(f"\nheld-out one-step action MAE (degrees):")
print(f"  BC policy    {mae_bc:.3f}")
print(f"  hold still   {mae_hold:.3f}   <- the baseline that makes the number mean something")
vram("after BC")
```

    BC (1 action)   275,462 params, trained in 2s on 9,060 windows

    held-out one-step action MAE (degrees):
      BC policy    1.670
      hold still   3.050   <- the baseline that makes the number mean something
    VRAM after BC              0.02 GB allocated /  0.05 GB reserved

``` python
from scipy.spatial import cKDTree

# Nearest-neighbour retrieval: no training, just "what did the human do last time the arm
# looked like this?". If a network cannot beat it, the network is a slower lookup table.
tree = cKDTree(run1["train_X"])
_, nn_idx = tree.query(run1["test_Xn"], k=1)
pred_nn = run1["train_Y"][nn_idx][:, 0]
mae_nn = float(np.abs(pred_nn - run1["true"][:, 0]).mean())

print(f"  nearest neighbour {mae_nn:.3f}")
print(f"\nPer-joint one-step MAE (degrees):")
print(f"{'joint':18s} {'BC':>8s} {'hold':>8s} {'1-NN':>8s}")
for j, name in enumerate(JOINTS):
    print(f"{name:18s} "
          f"{np.abs(run1['pred'][:, 0, j] - run1['true'][:, 0, j]).mean():8.3f} "
          f"{np.abs(run1['last_state'][:, j] - run1['true'][:, 0, j]).mean():8.3f} "
          f"{np.abs(pred_nn[:, j] - run1['true'][:, 0, j]).mean():8.3f}")
print("\nTwo things to notice. Nearest neighbour is the WORST of the three, badly so on "
      "shoulder_pan:\nwith 40 episodes the nearest training state can be from a different "
      "phase of a different\nepisode, and copying its action is then actively wrong. That "
      "is what the network buys you -\ninterpolation between demonstrations rather than "
      "retrieval of one.\n\nAnd the gripper is the interesting column: it is effectively "
      "binary (open or closed), so\nregressing its mean position is the wrong model - "
      "see section 9.")
free_memory()
```

      nearest neighbour 4.974

    Per-joint one-step MAE (degrees):
    joint                    BC     hold     1-NN
    shoulder_pan.pos      2.661    5.163   12.657
    shoulder_lift.pos     2.264    3.200    4.425
    elbow_flex.pos        1.546    3.630    4.109
    wrist_flex.pos        1.204    1.828    3.286
    wrist_roll.pos        1.242    2.206    3.770
    gripper.pos           1.104    2.273    1.599

    Two things to notice. Nearest neighbour is the WORST of the three, badly so on shoulder_pan:
    with 40 episodes the nearest training state can be from a different phase of a different
    episode, and copying its action is then actively wrong. That is what the network buys you -
    interpolation between demonstrations rather than retrieval of one.

    And the gripper is the interesting column: it is effectively binary (open or closed), so
    regressing its mean position is the wrong model - see section 9.

## 9. The Multimodality Problem

This is the single most important failure mode in imitation learning, and it is the reason Diffusion Policy exists.

**The setup.** A human demonstrating a task does not produce a function from state to action. At the same state, on different days, they go left or right around the obstacle; they close the gripper now or two frames later. The demonstrations are samples from a **multimodal conditional distribution** $p(a \mid s)$, not evaluations of a function.

**What MSE regression does with that.** It predicts the conditional **mean**. If half the demonstrations go left and half go right, the mean goes straight into the obstacle. L1 regression predicts the conditional median, which is better - it picks one mode instead of averaging - but is still a single point where a distribution was needed.

The cell measures this on **PushT**, the 2D pushing task Diffusion Policy was introduced on. For each state, it finds all other frames where the agent was in nearly the same place, and looks at how much the recorded actions disagree. **That disagreement is an error floor no deterministic policy can go below**, and comparing it to a trained policy’s error shows how much of the loss is irreducible.

PushT has a second, related lesson built in: its `observation.state` is only the **2D position of the pusher**, while the T-shaped block being pushed is visible only in the camera image. A state-only policy is therefore solving a partially observed problem and *cannot* be correct - which is precisely why the real Diffusion Policy takes the image.

------------------------------------------------------------------------

``` python
pusht, pusht_info = load_lerobot("lerobot/pusht")
P_STATE = np.stack(pusht["observation.state"].to_numpy()).astype("float32")
P_ACTION = np.stack(pusht["action"].to_numpy()).astype("float32")
P_EP = pusht["episode_index"].to_numpy()

print(f"pusht: {pusht_info['total_episodes']} episodes, {len(P_STATE):,} frames, "
      f"state {P_STATE.shape[1]}-D, action {P_ACTION.shape[1]}-D")
print(f"observation.state names: {pusht_info['features']['observation.state']['names']}")
print(f"the block's pose is NOT in the state - it is only in "
      f"{[k for k in pusht_info['features'] if 'image' in k]}")

# For each of a sample of states, find every frame within RADIUS pixels and measure how
# much the recorded actions disagree. That disagreement is irreducible for any policy
# that maps state to a single action.
RADIUS = 2.0
ptree = cKDTree(P_STATE)
probe = np.linspace(0, len(P_STATE) - 1, 2000).astype(int)
neighbourhoods = ptree.query_ball_point(P_STATE[probe], r=RADIUS)

spreads, sizes = [], []
for group in neighbourhoods:
    if len(group) >= 5:
        acts = P_ACTION[group]
        spreads.append(float(np.linalg.norm(acts - acts.mean(0), axis=1).mean()))
        sizes.append(len(group))

spreads = np.array(spreads)
print(f"\n{len(spreads)} probe states had >= 5 neighbours within {RADIUS:.0f} px "
      f"(median neighbourhood {int(np.median(sizes))} frames)")
print(f"mean distance from the group's own mean action: {spreads.mean():.2f} px")
print(f"90th percentile: {np.percentile(spreads, 90):.2f} px")
print(f"\nThat is the error floor. A deterministic policy asked for ONE action at these "
      f"states\ncannot do better on average, no matter how large the network.")
```

    pusht: 206 episodes, 25,650 frames, state 2-D, action 2-D
    observation.state names: {'motors': ['motor_0', 'motor_1']}
    the block's pose is NOT in the state - it is only in ['observation.image']

    686 probe states had >= 5 neighbours within 2 px (median neighbourhood 6 frames)
    mean distance from the group's own mean action: 16.99 px
    90th percentile: 29.57 px

    That is the error floor. A deterministic policy asked for ONE action at these states
    cannot do better on average, no matter how large the network.

``` python
# Train a state-only BC policy on PushT and put its error next to the floor.
p_mu, p_sd = P_STATE.mean(0), P_STATE.std(0) + 1e-6
a_mu, a_sd = P_ACTION.mean(0), P_ACTION.std(0) + 1e-6
p_train = P_EP < int(0.8 * (P_EP.max() + 1))

torch.manual_seed(0)
pusht_policy = nn.Sequential(nn.Linear(2, 256), nn.ReLU(),
                             nn.Linear(256, 256), nn.ReLU(),
                             nn.Linear(256, 2)).to(device)
opt = torch.optim.AdamW(pusht_policy.parameters(), lr=1e-3)
xt = torch.tensor((P_STATE[p_train] - p_mu) / p_sd, device=device)
yt = torch.tensor((P_ACTION[p_train] - a_mu) / a_sd, device=device)
for step in range(3000):
    i = torch.randint(0, len(xt), (512,), device=device)
    loss = F.mse_loss(pusht_policy(xt[i]), yt[i])
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.inference_mode():
    p_pred = pusht_policy(torch.tensor((P_STATE[~p_train] - p_mu) / p_sd,
                                       device=device)).cpu().numpy() * a_sd + a_mu
policy_err = float(np.linalg.norm(p_pred - P_ACTION[~p_train], axis=1).mean())

print(f"state-only BC on PushT: held-out action error {policy_err:.2f} px")
print(f"irreducible multimodality floor:               {spreads.mean():.2f} px")
print(f"\n{spreads.mean() / policy_err:.0%} of the policy's error is the multimodality floor, "
      f"not a modelling failure.\nMore capacity or more training cannot recover it. The fixes "
      f"are structural: give the policy the\nimage (so it can see the block), and model the "
      f"distribution instead of a point - which is\nDiffusion Policy in one sentence.")
del pusht_policy, opt
free_memory()
```

    state-only BC on PushT: held-out action error 21.83 px
    irreducible multimodality floor:               16.99 px

    78% of the policy's error is the multimodality floor, not a modelling failure.
    More capacity or more training cannot recover it. The fixes are structural: give the policy the
    image (so it can see the block), and model the distribution instead of a point - which is
    Diffusion Policy in one sentence.

``` python
from pyecharts.charts import Scatter

# One concrete neighbourhood: the pusher is in essentially the same place, and the human
# went several different directions. Nothing is wrong with the data - the task is
# genuinely multimodal, and a single predicted point has to sit in the middle of these.
best = int(np.argmax([len(g) if len(g) >= 5 else 0 for g in neighbourhoods]))
group = neighbourhoods[best]
acts = P_ACTION[group]
centre = P_STATE[probe[best]]

multimodal = (
    Scatter()
    .add_xaxis([round(float(v), 1) for v in acts[:, 0]])
    .add_yaxis("recorded actions", [[round(float(a[0]), 1), round(float(a[1]), 1)] for a in acts],
               symbol_size=9, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("their mean (what MSE predicts)",
               [[round(float(acts[:, 0].mean()), 1), round(float(acts[:, 1].mean()), 1)]],
               symbol="diamond", symbol_size=22, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("the pusher's position",
               [[round(float(centre[0]), 1), round(float(centre[1]), 1)]],
               symbol="triangle", symbol_size=18, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"PushT: {len(group)} frames where the pusher was within {RADIUS:.0f} px",
            subtitle="the human went several ways; the mean of those ways is not one of them"),
        xaxis_opts=opts.AxisOpts(type_="value", name="action x (px)"),
        yaxis_opts=opts.AxisOpts(type_="value", name="action y (px)"),
        tooltip_opts=opts.TooltipOpts(trigger="item"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
multimodal.render_notebook()
```

## 10. Action Chunking: the fix that made cheap hardware work

The second structural failure of naive BC is **compounding error**. A one-step policy is queried 30 times a second, and every query is a fresh chance to drift off the demonstrated distribution. Because the policy only ever saw expert states, its error grows with how far it has drifted, so small errors feed each other. Ross and Bagnell (2010) showed the drift is quadratic in the episode length; anyone who has watched a BC policy work perfectly for two seconds and then wander off has seen the theorem.

**ACT’s answer (Zhao et al., 2023): predict** $K$ actions at once and execute them open-loop. This helps in three separate ways:

- **Fewer decision points.** Executing a 16-step chunk means 16x fewer opportunities to compound.
- **Temporal consistency.** A chunk is internally coherent, so the motion is smooth. Independent per-step predictions produce jitter, which on real hardware is audible and wears out gears.
- **It absorbs inference latency.** A model at 5 Hz can drive a 30 Hz arm if each inference produces 6+ actions - which is exactly how a 3B-parameter VLA controls a real robot.

The cost is reduced reactivity: mid-chunk, the policy is blind. Real systems either re-plan every few steps or use **temporal ensembling** (average the overlapping predictions from several chunks), which ACT introduced.

The cell trains chunk sizes 1, 4, 16 and 32 on the SO-101 data and measures held-out error **per horizon step**. Two things to read off it.

First, the comparison against “hold still” at the same horizon. The policy’s margin **grows with the horizon** - about 1.5x better at the immediate next action, and more than 2x better a second out. Predicting where the arm is in 33 ms is nearly free; predicting where it will be in a second requires knowing what the task is.

Second, the cost: a larger chunk makes the **first** action slightly *worse*, because the same network capacity is now split across 32 outputs instead of one. That is the trade, stated plainly - you give up a little immediate precision to buy a coherent second of motion, and on real hardware that is a good trade because the coherence is what stops the arm from shaking itself apart.

------------------------------------------------------------------------

``` python
CHUNKS = [1, 4, 16, 32]
chunk_runs = {}
for k in CHUNKS:
    _, run = train_bc(chunk_size=k)
    chunk_runs[k] = run
    first = float(np.abs(run["pred"][:, 0] - run["true"][:, 0]).mean())
    whole = float(np.abs(run["pred"] - run["true"]).mean())
    print(f"chunk {k:2d}: {run['params']:>7,} params  {run['seconds']:4.0f}s   "
          f"first-action MAE {first:.3f}   whole-chunk MAE {whole:.3f}")

memory_report("after chunk sweep")
```

    chunk  1: 275,462 params     2s   first-action MAE 1.670   whole-chunk MAE 1.670
    chunk  4: 284,696 params     2s   first-action MAE 1.822   whole-chunk MAE 2.425
    chunk 16: 321,632 params     2s   first-action MAE 2.148   whole-chunk MAE 4.558
    chunk 32: 370,880 params     2s   first-action MAE 2.220   whole-chunk MAE 6.976
    RAM  after chunk sweep    15.67 / 20.97 GB
    VRAM after chunk sweep     0.02 GB allocated /  0.06 GB reserved

``` python
# Error against how far ahead the prediction reaches, for the largest chunk, next to
# the hold-still baseline at the same horizon.
run = chunk_runs[32]
horizons = np.arange(run["pred"].shape[1])
bc_by_h = np.abs(run["pred"] - run["true"]).mean(axis=(0, 2))
hold_by_h = np.abs(run["last_state"][:, None, :] - run["true"]).mean(axis=(0, 2))

horizon_line = (
    Line()
    .add_xaxis([f"{h / FPS * 1000:.0f}" for h in horizons])
    .add_yaxis("BC (chunk of 32)", [round(float(v), 3) for v in bc_by_h],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("hold still", [round(float(v), 3) for v in hold_by_h],
               is_smooth=True, symbol="none",
               linestyle_opts=opts.LineStyleOpts(type_="dashed"),
               label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Prediction error against how far ahead it reaches",
            subtitle=f"SO-101 held-out episodes; 32 steps at {FPS} Hz is about "
                     f"{32 / FPS:.1f} s of open-loop motion"),
        xaxis_opts=opts.AxisOpts(name="milliseconds ahead",
                                 axislabel_opts=opts.LabelOpts(interval=3)),
        yaxis_opts=opts.AxisOpts(name="mean absolute error (degrees)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
print(f"at 0 ms:   BC {bc_by_h[0]:.2f} vs hold-still {hold_by_h[0]:.2f} "
      f"({hold_by_h[0] / bc_by_h[0]:.2f}x)")
print(f"at {31 / FPS * 1000:.0f} ms: BC {bc_by_h[-1]:.2f} vs hold-still {hold_by_h[-1]:.2f} "
      f"({hold_by_h[-1] / bc_by_h[-1]:.2f}x)")
print("\nThe policy's advantage grows with the horizon: predicting the immediate next "
      "position is\nnearly free, and predicting where the arm will be in a second is "
      "where the task knowledge is.")
horizon_line.render_notebook()
```

    at 0 ms:   BC 2.22 vs hold-still 3.33 (1.50x)
    at 1033 ms: BC 9.69 vs hold-still 22.53 (2.33x)

    The policy's advantage grows with the horizon: predicting the immediate next position is
    nearly free, and predicting where the arm will be in a second is where the task knowledge is.

## 11. Head-to-head Benchmark

Every policy on the same held-out episodes, the same metric, the same normalisation. Two things to keep in front of you while reading it.

**This is action-prediction error, not success rate.** It measures agreement with the demonstrator. A policy that recovers from a slip differently from the human scores badly here and would score well on a robot; a policy with low error that fails at the moment of contact scores well here and fails on a robot. Closing that gap needs a simulator or hardware, which section 13 points at.

**The split is by episode.** Ten complete episodes the policy has never seen, not ten thousand frames sampled from the middle of episodes it has.

Hardware: knowledge-lab, RTX 3060 and 4 vCPU; 40 training episodes and 10 held-out ones (roughly 9,000 training windows, varying slightly with the chunk size).

------------------------------------------------------------------------

``` python
rows = [
    dict(policy="hold still (a_t = s_t)", first_step_mae=mae_hold, params=0, train_s=0.0),
    dict(policy="1-nearest neighbour", first_step_mae=mae_nn, params=0, train_s=0.0),
]
for k, run in chunk_runs.items():
    rows.append(dict(policy=f"BC, chunk of {k}",
                     first_step_mae=float(np.abs(run["pred"][:, 0] - run["true"][:, 0]).mean()),
                     params=run["params"], train_s=run["seconds"]))

bench = pd.DataFrame(rows).sort_values("first_step_mae").reset_index(drop=True)
bench["vs_hold_still"] = mae_hold / bench["first_step_mae"]
memory_report("after benchmark")
bench.round(3)
```

    RAM  after benchmark      15.67 / 20.97 GB
    VRAM after benchmark       0.02 GB allocated /  0.06 GB reserved

|     | policy                 | first_step_mae | params | train_s | vs_hold_still |
|-----|------------------------|----------------|--------|---------|---------------|
| 0   | BC, chunk of 1         | 1.670          | 275462 | 1.765   | 1.826         |
| 1   | BC, chunk of 4         | 1.822          | 284696 | 1.698   | 1.674         |
| 2   | BC, chunk of 16        | 2.148          | 321632 | 1.713   | 1.420         |
| 3   | BC, chunk of 32        | 2.220          | 370880 | 1.656   | 1.374         |
| 4   | hold still (a_t = s_t) | 3.050          | 0      | 0.000   | 1.000         |
| 5   | 1-nearest neighbour    | 4.974          | 0      | 0.000   | 0.613         |

``` python
from pyecharts.charts import Bar

# Two horizons, because they rank the policies differently - which is the point of
# section 10. Chunking costs a little at the first step and wins by a lot further out.
names = [f"chunk {k}" for k in CHUNKS]
first = [round(float(np.abs(chunk_runs[k]["pred"][:, 0] - chunk_runs[k]["true"][:, 0]).mean()), 3)
         for k in CHUNKS]
far = [round(float(np.abs(chunk_runs[k]["pred"][:, min(15, k - 1)]
                          - chunk_runs[k]["true"][:, min(15, k - 1)]).mean()), 3)
       for k in CHUNKS]

chunk_bar = (
    Bar()
    .add_xaxis(names)
    .add_yaxis("error at the first action", first)
    .add_yaxis(f"error {min(15, max(CHUNKS) - 1) / FPS * 1000:.0f} ms ahead", far)
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="What chunk size buys",
            subtitle="a larger chunk trades a little immediate accuracy for a lot of "
                     "coherence further ahead"),
        xaxis_opts=opts.AxisOpts(name="chunk size"),
        yaxis_opts=opts.AxisOpts(name="MAE (degrees)"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
chunk_bar.render_notebook()
```

``` python
# The predicted trajectory against the recorded one for a held-out episode. This is the
# plot that tells you whether a policy learned the task or the average pose.
run = chunk_runs[16]
test_eps = np.setdiff1d(np.unique(EPISODE), TRAIN_EPISODES)
_, _, window_eps = build_windows(16)
in_ep = np.where(window_eps[~np.isin(window_eps, TRAIN_EPISODES)] == test_eps[0])[0]

JOINT_SHOWN = JOINTS.index("gripper.pos") if "gripper.pos" in JOINTS else len(JOINTS) - 1
steps = list(range(len(in_ep)))

rollout_line = (
    Line()
    .add_xaxis([round(s / FPS, 2) for s in steps])
    .add_yaxis("recorded (human)",
               [round(float(run["true"][i, 0, JOINT_SHOWN]), 2) for i in in_ep],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("BC prediction (first action of each chunk)",
               [round(float(run["pred"][i, 0, JOINT_SHOWN]), 2) for i in in_ep],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"Held-out episode {test_eps[0]}: {JOINTS[JOINT_SHOWN]}",
            subtitle="the gripper is close to binary, so a regression policy rounds its "
                     "transitions - the classic BC artefact"),
        xaxis_opts=opts.AxisOpts(type_="value", name="seconds"),
        yaxis_opts=opts.AxisOpts(name="position"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
del policy1
free_memory()
vram("after policies freed")
rollout_line.render_notebook()
```

    VRAM after policies freed  0.02 GB allocated /  0.03 GB reserved

## 12. Grounding an Instruction: the perception front-end, live

Every VLA in section 6 has the same front half: turn a natural-language instruction and a camera image into a **target in the robot’s coordinate frame**. That half is fully runnable here through plain `transformers`, and it is worth building on its own because it is also the part you can debug without a robot.

The pipeline:

1.  **OWLv2** (`google/owlv2-base-patch16-ensemble`) - open-vocabulary detection. It takes free-text queries, so “the red mug” is a valid detector with no training. This is what lets a robot be told about an object it was never trained on.
2.  **Depth Anything V2-Small** - monocular relative depth for the whole frame. Sampling it inside the detected box gives the object’s distance.
3.  **Back-projection** - with a pinhole camera model, the pixel centre plus depth becomes a 3D point in camera coordinates. On a real robot one more fixed transform (hand-eye calibration) puts it in the arm’s frame; that matrix is measured once per robot and is the boring, essential step nobody demos.

**Honest limitation, stated once:** Depth Anything V2 produces **relative** depth, not metres. The values are consistent within a frame and unscaled between frames, so this gives a correct *bearing* and a correct *ordering* of candidates, not a metric grasp point. Real systems either use a stereo/RGB-D camera, a metric-depth checkpoint, or a known object size to fix the scale.

The cell tries the webcam first. If the camera sees nothing matching the instruction - the normal outcome for a lens pointed at a wall - it says so and falls back to a fixed scene, because “the object is not here” is a real branch a robot has to handle and not a reason to fake a detection.

------------------------------------------------------------------------

``` python
def require(*names):
    "Fail early and clearly if the notebook's setup / helper cells have not been run."
    missing = [n for n in names if n not in globals()]
    if missing:
        raise NameError(
            f"this demo needs {', '.join(missing)} from earlier in the notebook. "
            "Run the setup and helper cells first (Run > Run All Above Selected Cell)."
        )


require("device", "HF_CACHE", "DATA_DIR", "free_memory", "vram")

import urllib.request

import cv2
from PIL import Image
from transformers import pipeline

# --- get a frame ---------------------------------------------------------------------
# V4L2 backend + MJPEG + a warm-up read; do NOT set CAP_PROP_BUFFERSIZE (it halves the
# frame rate without making frames fresher). See dl-live-capture.md.
INSTRUCTION = "pick up the cup"
CANDIDATES = ["a cup", "a bottle", "a keyboard", "a hand", "a phone", "a book"]


def grab_webcam_frame(index=0, width=1280, height=720):
    "Return one RGB frame from /dev/videoN, or None if there is no usable camera."
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(5):                      # warm-up: the first frames are dark or stale
        ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def fallback_scene():
    "A fixed image with known objects, for when there is no camera or nothing in view."
    sample = DATA_DIR / "sample_scene.jpg"
    if not sample.exists():
        urllib.request.urlretrieve(
            "http://images.cocodataset.org/val2017/000000039769.jpg", sample)
    return (Image.open(sample).convert("RGB"),
            ["a cat", "a remote control", "a blanket", "a cup"],
            "pick up the remote control")


frame = grab_webcam_frame()
if frame is not None:
    image = Image.fromarray(frame)
    source = "the webcam at /dev/video0"
else:
    # No camera: say so, and use a fixed scene rather than pretending to have one.
    image, CANDIDATES, INSTRUCTION = fallback_scene()
    source = ("a downloaded sample image - NO CAMERA FOUND at /dev/video0. On the "
              "knowledge-lab LXC the video nodes are passed in from the Proxmox host by "
              "`av_devices` in infra/proxmox/variables.tf.")

print(f"instruction: {INSTRUCTION!r}")
print(f"frame {image.size} from {source}")
print(f"candidate labels: {CANDIDATES}")
```

    instruction: 'pick up the cup'
    frame (1280, 720) from the webcam at /dev/video0
    candidate labels: ['a cup', 'a bottle', 'a keyboard', 'a hand', 'a phone', 'a book']

``` python
# --- 1. open-vocabulary detection: language -> a box, with no task-specific training ---
detector = pipeline("zero-shot-object-detection",
                    model="google/owlv2-base-patch16-ensemble",
                    device=device, model_kwargs={"cache_dir": HF_CACHE})

THRESHOLD = 0.10        # open-vocabulary scores are not calibrated; 0.1 is a usable floor

t0 = time.perf_counter()
detections = detector(image, candidate_labels=CANDIDATES, threshold=THRESHOLD)
detect_s = time.perf_counter() - t0
print(f"OWLv2 on {source.split(' -')[0]}: {len(detections)} detections above "
      f"{THRESHOLD} in {detect_s:.2f}s")

# An empty frame is the normal case for a webcam pointed at a wall, and a robot has to
# handle it: "the object you named is not here" is a valid and important answer. The
# demo says so and then switches to a scene that does contain something, so the rest of
# the pipeline has an input - it does not invent a detection.
if not detections:
    print(f"\nNothing matching {CANDIDATES} is in front of the camera.")
    print("On a robot this is the branch that triggers a search behaviour or asks the "
          "operator.\nHere it falls back to the sample scene so sections 2 and 3 have "
          "something to ground.")
    image, CANDIDATES, INSTRUCTION = fallback_scene()
    source = "the fallback sample image (nothing was in the camera's view)"
    detections = detector(image, candidate_labels=CANDIDATES, threshold=THRESHOLD)
    print(f"\nre-ran on the sample scene: {len(detections)} detections")

for d in sorted(detections, key=lambda d: -d["score"])[:6]:
    print(f"  {d['score']:.3f}  {d['label']:20s} {d['box']}")

del detector
free_memory()
vram("after detection")
```

    OWLv2 on the webcam at /dev/video0: 0 detections above 0.1 in 2.37s

    Nothing matching ['a cup', 'a bottle', 'a keyboard', 'a hand', 'a phone', 'a book'] is in front of the camera.
    On a robot this is the branch that triggers a search behaviour or asks the operator.
    Here it falls back to the sample scene so sections 2 and 3 have something to ground.

    re-ran on the sample scene: 7 detections
      0.794  a remote control     {'xmin': 40, 'ymin': 73, 'xmax': 175, 'ymax': 117}
      0.706  a remote control     {'xmin': 334, 'ymin': 77, 'xmax': 368, 'ymax': 187}
      0.690  a cat                {'xmin': 6, 'ymin': 51, 'xmax': 329, 'ymax': 476}
      0.632  a cat                {'xmin': 341, 'ymin': 22, 'xmax': 640, 'ymax': 370}
      0.337  a blanket            {'xmin': 3, 'ymin': 92, 'xmax': 637, 'ymax': 479}
      0.247  a blanket            {'xmin': 6, 'ymin': 1, 'xmax': 641, 'ymax': 481}
    VRAM after detection       0.02 GB allocated /  0.03 GB reserved

``` python
# --- 2. monocular depth for the whole frame -------------------------------------------
depth_model = pipeline("depth-estimation",
                       model="depth-anything/Depth-Anything-V2-Small-hf",
                       device=device, model_kwargs={"cache_dir": HF_CACHE})

t0 = time.perf_counter()
depth_out = depth_model(image)
depth_s = time.perf_counter() - t0
depth = np.asarray(depth_out["predicted_depth"], dtype="float32")
print(f"Depth Anything V2-Small: {depth.shape} in {depth_s:.2f}s "
      f"(relative depth, range {depth.min():.1f} to {depth.max():.1f} - NOT metres)")

del depth_model
free_memory()

# --- 3. back-project the target into camera coordinates -------------------------------
# The instruction names the object; pick the highest-scoring matching detection.
target_word = INSTRUCTION.split("pick up the ")[-1].strip()
matches = [d for d in detections if target_word.split()[-1] in d["label"]]
if not matches and not detections:
    raise RuntimeError(
        "no detections to ground - point the camera at one of "
        f"{CANDIDATES}, or lower THRESHOLD in the cell above.")
if not matches:
    print(f"nothing labelled {target_word!r}; grounding the highest-scoring "
          f"detection instead")
target = max(matches or detections, key=lambda d: d["score"])

box = target["box"]
cx = (box["xmin"] + box["xmax"]) / 2
cy = (box["ymin"] + box["ymax"]) / 2

# Depth is returned at the image resolution here; sample a small patch at the centre and
# take the median, which is robust to a stray edge pixel inside the box.
h, w = depth.shape
py, px = int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))
patch = depth[max(0, py - 5): py + 6, max(0, px - 5): px + 6]
z_rel = float(np.median(patch))

# Pinhole back-projection. fx/fy come from camera calibration; this uses a plausible
# 60-degree horizontal field of view, which is a guess, not a measurement.
FOV_DEG = 60.0
fx = fy = (image.width / 2) / np.tan(np.radians(FOV_DEG) / 2)
x_cam = (cx - image.width / 2) / fx
y_cam = (cy - image.height / 2) / fy

print(f"\ntarget: {target['label']!r} at {target['score']:.3f} confidence")
print(f"  pixel centre     ({cx:.0f}, {cy:.0f}) of {image.width}x{image.height}")
print(f"  relative depth   {z_rel:.1f}  (larger = nearer for this model)")
print(f"  bearing from the optical axis: "
      f"{np.degrees(np.arctan(x_cam)):+.1f} deg horizontal, "
      f"{np.degrees(np.arctan(y_cam)):+.1f} deg vertical")
print("\nOn a real arm the next line is a fixed 4x4 hand-eye transform T_base_camera,")
print("measured once with a calibration board, turning this into a joint-space goal.")
print("What is missing here is the SCALE: relative depth gives direction, not metres.")
```

    Depth Anything V2-Small: (480, 640) in 0.15s (relative depth, range 0.8 to 5.2 - NOT metres)

    target: 'a remote control' at 0.794 confidence
      pixel centre     (108, 95) of 640x480
      relative depth   1.7  (larger = nearer for this model)
      bearing from the optical axis: -21.0 deg horizontal, -14.7 deg vertical

    On a real arm the next line is a fixed 4x4 hand-eye transform T_base_camera,
    measured once with a calibration board, turning this into a joint-space goal.
    What is missing here is the SCALE: relative depth gives direction, not metres.

``` python
from pyecharts.charts import HeatMap

# The depth map, downsampled, with the target marked. The point of the picture is to
# check that the detection landed on something that is actually at a sensible distance -
# the most common failure is a confident box on a background object.
STRIDE = max(1, depth.shape[1] // 96)
small = depth[::STRIDE, ::STRIDE]
sh, sw = small.shape
cells = [[c, sh - 1 - r, round(float(small[r, c]), 1)] for r in range(sh) for c in range(sw)]

depth_map = (
    HeatMap()
    .add_xaxis([str(c) for c in range(sw)])
    .add_yaxis("relative depth", [str(r) for r in range(sh)], cells,
               label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Monocular depth, with the grounded target",
            subtitle=f"{target['label']!r} sits at column {int(cx / STRIDE)}, "
                     f"row {sh - 1 - int(cy / STRIDE)} of this grid"),
        visualmap_opts=opts.VisualMapOpts(min_=float(small.min()), max_=float(small.max()),
                                          is_calculable=True, orient="horizontal",
                                          pos_bottom="2%"),
        xaxis_opts=opts.AxisOpts(name="image column (downsampled)",
                                 axislabel_opts=opts.LabelOpts(interval=9)),
        yaxis_opts=opts.AxisOpts(name="image row",
                                 axislabel_opts=opts.LabelOpts(interval=9)),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
)
free_memory()
memory_report("after perception")
depth_map.render_notebook()
```

    RAM  after perception     16.28 / 20.97 GB
    VRAM after perception      0.02 GB allocated /  0.03 GB reserved

## 13. Common Frameworks

Robot learning is the task in this folder with a hardware dependency, and that changes the shape of the ecosystem completely. Half of it is simulators, because you cannot get a number with an error bar on a physical arm; a large part is data tooling, because demonstrations are recorded video plus joint states rather than files you download; and the model libraries are consolidating around one project, LeRobot, that ships policies, datasets and hardware drivers together.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [LeRobot](https://github.com/huggingface/lerobot) | modelling | ACT, Diffusion Policy, pi0 and SmolVLA behind one training script, the `LeRobotDataset` loader that reads the video streams, and drivers for the SO-100/SO-101 arms | Apache 2.0 | Actually building a robot. A vendor runtime rather than a general-purpose library, which is why section 7 reads the parquet directly - but it is the right tool |
| [transformers](https://github.com/huggingface/transformers) | modelling | The VLA backbones (SmolVLA, and the vision-language halves of the larger policies) and the perception front end of section 12 | Apache 2.0 | Language grounding, and fine-tuning a VLA - which inherits knowledge 50 demonstrations cannot provide |
| [diffusers](https://github.com/huggingface/diffusers) | modelling | The denoising machinery behind Diffusion Policy - the direct fix for the multimodality measured in section 9 | Apache 2.0 | Your demonstrations contain more than one valid way to do the task, which is nearly always |
| [MuJoCo](https://github.com/google-deepmind/mujoco) / [robosuite](https://github.com/ARISE-Initiative/robosuite) / [ManiSkill](https://github.com/haosulab/ManiSkill) | data | Fast contact physics and manipulation task suites you can evaluate in overnight, thousands of times | Apache 2.0 / MIT | Always before hardware. This is the only way to get an error bar, and the only way to compare with published results |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | data | GPU-parallel simulation with photorealistic rendering, for domain randomisation and sim-to-real transfer | BSD-3 (Isaac Sim is proprietary) | Sim-to-real. Randomising textures, lighting and dynamics is what makes a simulator-trained policy survive contact with a real camera |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) / [CALVIN](https://github.com/mees/calvin) | data | Language-conditioned manipulation benchmarks with fixed task suites and published baselines | MIT | Evaluating a VLA. “It worked on my desk” is not a result anyone can compare against |
| [ONNX Runtime](https://onnxruntime.ai/) / [TensorRT](https://github.com/NVIDIA/TensorRT) | inference runtime | The policy exported to run on the robot’s onboard computer at a fixed control rate | MIT / Apache 2.0 (TensorRT SDK proprietary) | Deployment. A policy that misses its control deadline is a different policy, regardless of its offline score |
| [ROS 2](https://github.com/ros2) | orchestration | The middleware real robots run on: node graph, message passing, transforms, drivers, and the safety layer between the policy and the motors | Apache 2.0 | Any hardware beyond a single USB arm. Also where joint limits and the e-stop live, which is where they belong |
| [Weights & Biases](https://github.com/wandb/wandb) + rollout video | evaluation | Success rate across seeds and, critically, the recorded rollouts - because a failed grasp looks nothing like a bad loss curve | MIT | Every run. In this task the video *is* the metric that matters |

**The 2026 default stack** is LeRobot for policies and data, MuJoCo or ManiSkill for evaluation, a fine-tuned SmolVLA rather than a from-scratch policy, ROS 2 on the hardware, and success rate in simulation across seeds as the number you report. Cameras from the start - every real manipulation policy uses vision, because the object’s pose is not in the proprioception.

The common wrong turn is evaluating on held-out demonstration loss. Section 9 is the point: action-space MSE and task success are only loosely related, because averaging two valid approaches produces an invalid one. Evaluate by rolling out. The second is skipping the safety layer - joint and velocity limits in software, a workspace box, torque limits, a clamped policy output and a physical e-stop belong in place before the arm moves for the first time.

------------------------------------------------------------------------

## 14. Going Further

**Replace the MLP with a real architecture.** The step from section 10’s chunked MLP to **ACT** is adding a ResNet image encoder, a transformer, and a CVAE latent to absorb demonstrator variability. The step to **Diffusion Policy** is replacing the regression head with a denoising model, which is the direct fix for the multimodality measured in section 9. Both are a few hundred lines and both are in LeRobot.

**Add the cameras.** Every result here uses joint states only, and every real manipulation policy uses vision, because the object’s pose is not in the proprioception. Section 9 shows what that costs: on PushT, a state-only policy is at the irreducible floor and cannot improve.

**Evaluate in simulation before hardware.** LIBERO and CALVIN for language-conditioned manipulation, robosuite/ManiSkill for physics, Isaac Lab for GPU-parallel RL. It is the only way to get a number with a usable error bar, and the only way to compare against published results.

**Fine-tune a VLA rather than training from scratch.** SmolVLA at 450M fits on this box and fine-tunes from a few hundred episodes; pi0 (Apache 2.0 weights) is the stronger starting point if you have more VRAM. Both inherit language grounding you cannot get from 50 demonstrations.

**Take safety seriously before the arm moves.** Joint and velocity limits in software, a workspace box the end effector may not leave, torque limits, a physical e-stop, and a policy output that is clamped rather than trusted. A learned policy will occasionally command something absurd; the question is only whether the layer below it refuses.

**Related notebooks in this repo:** `00_Reinforcement_Learning` (the reward-driven half of robot learning, and the Decision Transformer that shares its offline framing), `Computer_Vision/00_Depth_Estimation` and `13_Zero_Shot_Object_Detection` (the two models section 12 chains together, each in depth), and `Multimodal/01_Image_Text_to_Text` (the VLM backbone every VLA is built on).

------------------------------------------------------------------------

Back to top
