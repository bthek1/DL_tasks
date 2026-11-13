# Reinforcement Learning

Learning to act from reward instead of labels: the MDP that every algorithm is solving, why the value-based and policy-gradient families split, how RLHF and GRPO turned RL into the standard finishing step for language models, and runnable code that trains Q-learning, REINFORCE and actor-critic from scratch and then runs a Decision Transformer on real offline data.

Author

Benedict Thekkel

## 1. What is Reinforcement Learning?

Reinforcement learning is the setting where **there are no labelled examples, only consequences**. An agent takes actions in an environment, the environment returns a scalar reward and a new state, and the agent has to work out a behaviour that collects a lot of reward over time. Nobody ever tells it what the right action was.

Formally, almost everything is a **Markov Decision Process** $(S, A, P, R, \gamma)$:

- $S$ - states. $A$ - actions (discrete, like “move left”, or continuous, like joint torques).
- $P(s' \mid s, a)$ - transition dynamics. **Model-free** methods never learn this; **model-based** methods do, and use it to plan.
- $R(s, a)$ - reward. $\gamma \in [0, 1)$ - discount, which makes an infinite sum finite and encodes how much the future matters.

The agent learns a **policy** $\pi(a \mid s)$ maximising the expected discounted return $G_t = \sum_{k \ge 0} \gamma^k r_{t+k}$. Two functions carry almost all the theory:

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s], \qquad
  Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$$

and they satisfy the **Bellman equation**, which is what every value-based algorithm is a stochastic approximation of:

$$Q^*(s, a) = \mathbb{E}\big[r + \gamma \max_{a'} Q^*(s', a')\big]$$

**What makes RL genuinely different from supervised learning:**

- **The data depends on the policy.** Change the policy and you change the distribution you learn from. Nothing is i.i.d., and the “training set” moves under you.
- **Credit assignment is delayed.** The move that lost the chess game happened forty moves before the loss.
- **Exploration versus exploitation.** You cannot learn about actions you never take, and taking bad actions costs reward now.
- **The reward is a specification, and specifications leak.** An agent optimises what you wrote, not what you meant - this is reward hacking, and it is the default outcome, not an edge case.

**Neighbouring tasks:**

| Task | What it does | Typical tool |
|----|----|----|
| Imitation learning / behaviour cloning | Copy demonstrations; supervised, no reward | see `01_Robotics` |
| Inverse RL | Infer the reward function from behaviour | MaxEnt IRL, GAIL |
| Offline RL | Learn from a fixed logged dataset, no interaction | CQL, IQL, Decision Transformer (section 12) |
| Bandits | One-step RL: no state transitions | Thompson sampling, UCB |
| RLHF / preference optimisation | Fine-tune an LLM from human or model preferences | PPO, DPO, GRPO (section 13) |
| Model predictive control | Plan with a known model, re-plan each step | Classical control, MPPI |

------------------------------------------------------------------------

## 2. Real-World Use Cases

RL has a reputation for being spectacular in games and disappointing in production. That was fair until roughly 2022; the language-model wave changed the balance decisively, and RL is now a routine step in the most widely deployed AI systems in the world.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| **LLM post-training (RLHF / RLVR)** | Every frontier lab | Prompts + preference or verifier reward -\> aligned model | **The single biggest deployment of RL today**; reward-model quality and KL control dominate |
| Recommendation and feed ranking | YouTube, Spotify, TikTok | User state -\> item; reward = long-term engagement | Off-policy evaluation (you cannot A/B test every policy); delayed and gameable reward |
| Datacentre and building cooling | Infrastructure (DeepMind/Google, 2018) | Sensor state -\> setpoints; reward = energy used | Safety constraints are hard limits; a bad action costs real hardware |
| Robot control and manipulation | Robotics | Proprioception + vision -\> torques | Sample efficiency; sim-to-real gap; safety - see `01_Robotics` |
| Chip floorplanning | EDA (Google, 2021) | Partial placement -\> next macro position | The reward is a slow simulator; enormous action space |
| Inventory, pricing and logistics | Retail, supply chain | Stock and demand state -\> order quantity | Non-stationary demand; classical operations research is a strong incumbent |
| Autonomous driving (planning layer) | Automotive | Scene state -\> trajectory | Safety certification; almost always offline RL or imitation, not online exploration |
| Algorithmic trading | Finance | Market state -\> position | Non-stationarity; a near-zero signal-to-noise ratio; catastrophic tail risk |
| Games | Research and entertainment | Board/pixels -\> move | Where the field’s landmarks happened: TD-Gammon, Atari, AlphaGo, AlphaStar, Gran Turismo Sophy |

**What the benchmark reward hides.**

- **Sample efficiency is the whole game outside a simulator.** Model-free RL routinely needs $10^6$ to $10^9$ environment steps. That is free in a simulator and impossible on a physical robot or a live recommender, which is why almost every real deployment is offline RL, imitation, or sim-to-real.
- **Reward hacking is the norm.** Agents find the specification’s holes: the classic CoastRunners boat that spins in a lagoon collecting powerups instead of finishing the race, the simulated robot that exploits a physics bug to “walk”. In RLHF the same failure looks like sycophancy and verbosity - the reward model prefers longer, more agreeable answers, so the policy produces them.
- **Reproducibility is poor.** Henderson et al. (2018) showed that with the same algorithm and different random seeds, published RL results can swing enough to change the ranking. Any RL number without multiple seeds and a confidence band should be read as an anecdote.
- **Distribution shift breaks offline RL specifically.** A Q-function queried on actions absent from the logged data extrapolates wildly and confidently, and the policy then chases that fantasy. Every serious offline method (CQL, IQL, TD3+BC) is a different way of refusing to answer questions the data cannot support.
- **Safety during exploration is often non-negotiable.** “Try a random action and see” is not available on a power grid, a patient, or a car. Constrained RL, shielding and simulation exist because of this.

------------------------------------------------------------------------

## 3. How Modern Reinforcement Learning Works

**1. Dynamic programming (Bellman, 1957).** With a known model, value iteration and policy iteration solve the MDP exactly. Nobody has the model, but every model-free algorithm is a sampled approximation of these updates.

**2. Temporal-difference learning (Sutton, 1988) and Q-learning (Watkins, 1989).** Learn from a single transition by bootstrapping off your own current estimate:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\big]$$

The bracket is the **TD error**. Q-learning is **off-policy** - it learns the greedy policy’s values while behaving differently (e.g. epsilon-greedy), which is what makes replay buffers legal. Section 8 implements this in about fifteen lines.

**3. Deep Q-Networks (Mnih et al., 2013/2015).** Replace the table with a CNN and Atari became tractable. The two tricks that made it stable are still standard: an **experience replay buffer** (breaks the correlation between consecutive samples) and a **target network** (stops the bootstrap target from moving every step). Rainbow (2017) bundled six further improvements.

**4. Policy gradients (Williams, 1992; Sutton et al., 2000).** Parameterise the policy directly and push up the log-probability of actions that did well:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi\big[\nabla_\theta \log \pi_\theta(a \mid s) \cdot G_t\big]$$

REINFORCE (section 9) is this exactly. It is unbiased and extremely high variance, which is why nobody uses it raw.

**5. Actor-critic and the advantage (2000 -\> now).** Subtract a learned baseline $V(s)$ from the return to get the **advantage** $A(s,a) = Q(s,a) - V(s)$: “how much better than usual was this action”. Same expected gradient, far less variance. A2C/A3C (2016), GAE (2016). Section 10 implements it and measures the variance reduction directly.

**6. Trust regions and PPO (2015-2017).** A big policy update can collapse performance irrecoverably, because the next batch of data is collected by the broken policy. TRPO constrains the KL divergence between old and new policy; **PPO** (Schulman et al., 2017) gets most of the benefit with a clipped surrogate objective and a few lines of code. **PPO is still the default on-policy algorithm in 2026**, including for RLHF.

**7. Off-policy continuous control (2015-2018).** DDPG, TD3, and **SAC** (soft actor-critic, which adds an entropy bonus to the objective so the policy stays stochastic and explores). SAC is the default for continuous control from scratch.

**8. Model-based RL (2018 -\> now).** Learn the dynamics and train inside the learned model. Dreamer v1-v3 (2019-2023) and MuZero (2019, which learns a model sufficient for planning without ever predicting pixels). **DreamerV3 (2023) was the first agent to collect diamonds in Minecraft from scratch with no human data**, using one hyperparameter set across more than 150 tasks.

**9. Offline RL and sequence modelling (2019 -\> now).** Learn from a fixed dataset. Either constrain the policy to the data (BCQ, CQL, IQL, TD3+BC), or drop the RL machinery entirely: **Decision Transformer** (Chen et al., 2021) treats a trajectory as a sequence of (return-to-go, state, action) tokens and just does autoregressive prediction, conditioning on the return you want at test time. Section 12 runs one.

**10. RL for language models (2022 -\> now), where RL actually became ubiquitous.**

- **RLHF** (InstructGPT, 2022): collect human preference comparisons, fit a reward model, optimise the policy with PPO under a KL penalty against the base model.
- **DPO** (2023): a closed-form reparameterisation that skips the reward model and optimises preferences directly with a supervised-looking loss. Much simpler, and it removed RL from a lot of pipelines.
- **RLVR / GRPO** (DeepSeekMath 2024, DeepSeek-R1 2025): when the reward is *verifiable* - the maths answer is right, the unit test passes - you do not need a reward model at all. **GRPO** drops the value network too, sampling a group of completions per prompt and using the group’s mean reward as the baseline. This is the recipe behind the 2025 reasoning models, and it is the most consequential thing to happen to RL in a decade.

------------------------------------------------------------------------

## 4. Evaluation Metrics

RL evaluation is unusually treacherous, because the number people quote (episode return) is a noisy sample from a distribution that depends on the seed, the exploration noise, and the environment version.

**Return.** The sum of rewards per episode, usually undiscounted for reporting even when training discounted. Report the **mean over many episodes and several training seeds, with an interval**.

**Normalised score.** Raw return is not comparable across environments, so benchmarks normalise:

$$\text{normalised} = 100 \times \frac{\text{score} - \text{score}_{\text{random}}}
                                       {\text{score}_{\text{expert}} - \text{score}_{\text{random}}}$$

This is what the D4RL offline benchmark and the Atari human-normalised scores use. 100 means expert-level; above 100 means better than the reference expert.

**Sample efficiency.** Return as a function of environment steps - the learning curve is the result, not the final number. Two algorithms reaching the same return in $10^5$ and $10^7$ steps are not comparable achievements.

**Regret.** Cumulative difference from the optimal policy’s return. The standard metric in bandits and the honest one when learning happens online in production.

**Statistical practice (Agarwal et al., NeurIPS 2021, “Deep RL at the Edge of the Statistical Precipice”).** The field’s reporting was bad enough to need a paper. Its recommendations are now standard and worth following:

- Report the **interquartile mean (IQM)** rather than the mean - a few lucky seeds move the mean a lot.
- Show **stratified bootstrap confidence intervals**, not standard deviations over 3 seeds.
- Use **performance profiles** (the fraction of runs above a score threshold) instead of a table of point estimates.

------------------------------------------------------------------------

``` python
import numpy as np

# Why the mean over seeds is not enough: 10 runs of a plausible RL algorithm, where
# 2 seeds diverged. This is normal, not pathological.
rng = np.random.default_rng(0)
returns = np.concatenate([rng.normal(320, 25, 8), rng.normal(40, 15, 2)])   # 2 failed seeds


def iqm(x):
    "Interquartile mean: drop the top and bottom 25%, then average."
    lo, hi = np.percentile(x, [25, 75])
    return float(x[(x >= lo) & (x <= hi)].mean())


def bootstrap_ci(x, stat=np.mean, n=10000, alpha=0.05, seed=0):
    "Percentile bootstrap confidence interval for a statistic."
    r = np.random.default_rng(seed)
    samples = [stat(r.choice(x, len(x), replace=True)) for _ in range(n)]
    return tuple(float(v) for v in np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


print(f"10 seeds: {np.sort(returns).round(0)}")
print(f"mean   {returns.mean():6.1f}   95% CI {tuple(round(v, 1) for v in bootstrap_ci(returns))}")
print(f"median {np.median(returns):6.1f}")
print(f"IQM    {iqm(returns):6.1f}   95% CI "
      f"{tuple(round(v, 1) for v in bootstrap_ci(returns, iqm))}")
print(f"\nThe mean is dragged down by two failures and its interval is enormous. Reporting "
      f"'our method\nscores {returns.mean():.0f}' hides both facts; the IQM says what the "
      f"method does when it works.")

# Normalised score, the D4RL / Atari convention.
random_score, expert_score = 20.0, 350.0
print(f"\nnormalised: {100 * (returns.mean() - random_score) / (expert_score - random_score):.1f} "
      f"(100 = the reference expert)")
```

    10 seeds: [ 21.  29. 307. 317. 323. 323. 329. 336. 344. 353.]
    mean    268.1   95% CI (179.2, 331.9)
    median  322.9
    IQM     322.9   95% CI (151.1, 338.0)

    The mean is dragged down by two failures and its interval is enormous. Reporting 'our method
    scores 268' hides both facts; the IQM says what the method does when it works.

    normalised: 75.2 (100 = the reference expert)

## 5. Environments and Datasets

RL “datasets” are mostly **environments** - simulators you interact with - plus, for offline RL, a fixed corpus of logged trajectories.

| Environment / dataset | Contents | Scale | Type | License | Typical use |
|----|----|----|----|----|----|
| [Gymnasium](https://gymnasium.farama.org/) | The standard API plus classic control, Box2D, MuJoCo, Atari | ~100 envs | Online | MIT | The default interface for everything |
| [MuJoCo / `gymnasium[mujoco]`](https://mujoco.org/) | Physics locomotion tasks (Hopper, Walker2d, HalfCheetah, Ant) | 10+ tasks | Online, continuous | Apache 2.0 | The continuous-control benchmark |
| [Atari / ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | 57 Atari 2600 games from pixels | 57 games | Online, discrete | GPL-2.0 (ALE) | The discrete-action benchmark since DQN |
| [D4RL](https://github.com/Farama-Foundation/D4RL) / [Minari](https://minari.farama.org/) | Logged trajectories at several quality levels (random/medium/expert) | ~10^6 transitions per task | **Offline** | Apache 2.0 | The offline-RL standard |
| [decision_transformer_gym_replay](https://huggingface.co/datasets/edbeeching/decision_transformer_gym_replay) | D4RL hopper/walker/halfcheetah as pickled trajectories | 2,186 trajectories (hopper-medium) | Offline | Apache 2.0 | **Used by this notebook** (section 12) |
| [ProcGen](https://github.com/openai/procgen) | 16 procedurally generated games | infinite levels | Online | MIT | Measuring generalisation, not memorisation |
| [MineRL / MineDojo](https://minerl.io/) | Minecraft, with human demonstrations | 60M+ frames | Both | MIT | Long-horizon, open-ended tasks |
| [PettingZoo](https://pettingzoo.farama.org/) | Multi-agent environments | ~60 envs | Online | MIT | Multi-agent RL |
| [Open X-Embodiment](https://robotics-transformer-x.github.io/) | Real robot trajectories, 22 embodiments | 1M+ episodes | Offline | CC BY 4.0 | Robot learning - see `01_Robotics` |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback), [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | LLM preference pairs | 64k / 170k comparisons | Offline preferences | MIT / MIT | RLHF and DPO training |

**This notebook does not install a simulator.** `gymnasium` and MuJoCo are not repo dependencies, and pulling in a physics engine to demonstrate the Bellman equation is the wrong trade. Instead:

- Sections 8-11 use a **gridworld written in about 30 lines of numpy** in this notebook. It is a real MDP with stochastic transitions, and it makes the algorithms legible in a way a black-box simulator does not.
- Section 12 uses **real D4RL data** (hopper-medium-v2, 2,186 logged trajectories from a MuJoCo Hopper) and a **real pretrained Decision Transformer**, both from the Hugging Face Hub. No simulator is needed to evaluate a model on logged data - which is exactly the point of offline RL.

------------------------------------------------------------------------

## 6. The Algorithm Landscape (mid-2026)

| Algorithm | Year | Family | Action space | On/off-policy | Sample efficiency | Best for |
|----|----|----|----|----|----|----|
| Q-learning (tabular) | 1989 | Value | discrete, small | off | \- | **Understanding the field** - built in section 8 |
| DQN / Rainbow | 2015/2017 | Value | discrete | off | medium | Atari-style discrete control |
| REINFORCE | 1992 | Policy gradient | any | on | very low | Pedagogy - built in section 9 |
| A2C / A3C | 2016 | Actor-critic | any | on | low | Simple parallel baseline - built in section 10 |
| **PPO** | 2017 | Actor-critic | any | on | medium | **The default. Robust, simple, used for RLHF** |
| DDPG / TD3 | 2015/2018 | Actor-critic | continuous | off | high | Continuous control |
| **SAC** | 2018 | Actor-critic + entropy | continuous | off | high | **The default for continuous control from scratch** |
| MuZero | 2019 | Model-based + search | discrete | off | very high | Games with a clear win condition |
| **DreamerV3** | 2023 | Model-based (world model) | any | off | very high | **One hyperparameter set across 150+ tasks; Minecraft diamonds from scratch** |
| CQL / IQL / TD3+BC | 2020-21 | Offline value | any | offline | n/a | Learning from logged data safely |
| **Decision Transformer** | 2021 | Offline sequence model | any | offline | n/a | **Offline RL as autoregressive prediction** - run in section 12 |
| **PPO for RLHF** | 2022 | Actor-critic | token | on | n/a | **InstructGPT-style alignment** |
| **DPO** | 2023 | Preference optimisation | token | offline | n/a | **RLHF without a reward model or RL loop** |
| **GRPO / RLVR** | 2024-25 | Policy gradient, no critic | token | on | n/a | **Verifiable rewards; the 2025 reasoning models** |

**Libraries.** `stable-baselines3` (the reliable reference implementations of PPO/SAC/TD3/DQN), `CleanRL` (single-file implementations, the best way to actually read an algorithm), `Ray RLlib` (distributed), and **`trl`** (Hugging Face: PPO, DPO, GRPO for language models, and the one most people will actually use in 2026).

**What wins what.** For continuous control from scratch, SAC and DreamerV3. For discrete action spaces with lots of simulation available, PPO. For anything where you cannot interact - logged data, real robots, live recommenders - offline RL or a Decision Transformer. For language models, DPO when you have preference pairs and GRPO when you have a verifier.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `numpy` - the gridworld environment and tabular Q-learning
- `torch` - REINFORCE, actor-critic, and the Decision Transformer
- `transformers` (\>=5.13) - `DecisionTransformerModel`, transformers-native, no vendor RL package
- `huggingface_hub` - fetching the D4RL trajectory pickle
- `pyecharts` - all charts (repo rule)

Memory: the gridworld is a 100-state table, the policy networks are a few thousand parameters, and the Decision Transformer is **0.86M parameters** - the entire notebook is a rounding error against the 12 GB card. The `free_memory()` discipline is still applied between sections because it is the house rule, and because the pattern is what transfers, not the number.

Downloads (the D4RL pickle, the DT checkpoint) land in `DL_tasks/datasets/`, which is gitignored.

------------------------------------------------------------------------

``` python
# No simulator dependency: the environment below is ~30 lines of numpy, and the
# offline section uses logged trajectories from the Hub.
# %pip install -q torch transformers huggingface_hub numpy pyecharts

# To run the same algorithms against real environments:
# %pip install -q "gymnasium[classic-control,box2d,mujoco]" stable-baselines3
```

``` python
import ctypes
import ctypes.util
import gc
import pickle
import time
from pathlib import Path

import numpy as np
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
        print(f"VRAM {tag:18s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")


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
    print(f"RAM  {tag:18s} {(vm.total - vm.available) / 1e9:5.2f} / {vm.total / 1e9:5.2f} GB")
    vram(tag)


DATA_DIR = Path("../../datasets")        # gitignored
DATA_DIR.mkdir(exist_ok=True)
HF_CACHE = str(DATA_DIR / "hf_cache")

torch.manual_seed(0)
memory_report("baseline")
```

    NVIDIA GeForce RTX 3060
    device: cuda:0
    RAM  baseline           14.85 / 20.97 GB
    VRAM baseline            0.00 GB allocated /  0.00 GB reserved

``` python
class WindyGrid:
    """A 10x10 gridworld with wind, a goal, and lava. Deliberately small and readable.

    It is a genuine MDP and not a toy in the dismissive sense: transitions are
    stochastic (the wind pushes the agent sideways with probability `wind`), the
    reward is sparse and delayed, and there is a shortcut past the lava that a
    risk-neutral optimal policy takes and a timid one does not.

        S . . . . . . . . .
        . . . # # # # . . .        # lava   (-50, episode ends)
        . . . . . . . . . .        G goal   (+100, episode ends)
        ...                        every step costs -1
        . . . . . . . . . G
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]      # up, down, left, right
    ACTION_NAMES = ["up", "down", "left", "right"]

    def __init__(self, size=10, wind=0.15, seed=0):
        self.size, self.wind = size, wind
        self.rng = np.random.default_rng(seed)
        self.lava = {(1, c) for c in range(3, 7)}
        self.start, self.goal = (0, 0), (size - 1, size - 1)
        self.n_states, self.n_actions = size * size, 4

    def reset(self):
        self.pos = self.start
        return self.state_id(self.pos)

    def state_id(self, pos):
        return pos[0] * self.size + pos[1]

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        # Wind: with probability `wind` the move is deflected 90 degrees.
        if self.rng.random() < self.wind:
            dr, dc = (0, self.rng.choice([-1, 1])) if dr != 0 else (self.rng.choice([-1, 1]), 0)
        r = int(np.clip(self.pos[0] + dr, 0, self.size - 1))
        c = int(np.clip(self.pos[1] + dc, 0, self.size - 1))
        self.pos = (r, c)

        if self.pos in self.lava:
            return self.state_id(self.pos), -50.0, True
        if self.pos == self.goal:
            return self.state_id(self.pos), 100.0, True
        return self.state_id(self.pos), -1.0, False       # step cost: hurry up


env = WindyGrid()
print(f"{env.n_states} states, {env.n_actions} actions, wind {env.wind:.0%}, "
      f"{len(env.lava)} lava cells")

# A random policy, for a floor to compare against.
def rollout(policy_fn, environment, max_steps=200, episodes=200):
    "Run episodes under a policy; return the array of undiscounted returns."
    out = []
    for _ in range(episodes):
        s, total, done, steps = environment.reset(), 0.0, False, 0
        while not done and steps < max_steps:
            s, r, done = environment.step(policy_fn(s))
            total += r
            steps += 1
        out.append(total)
    return np.array(out)


random_returns = rollout(lambda s: env.rng.integers(0, 4), env)
print(f"\nrandom policy: mean return {random_returns.mean():7.1f} "
      f"+/- {random_returns.std():.1f}   "
      f"reached the goal in {(random_returns > 0).mean():.1%} of episodes")
```

    100 states, 4 actions, wind 15%, 4 lava cells

    random policy: mean return   -94.0 +/- 51.7   reached the goal in 2.5% of episodes

## 8. Tabular Q-Learning

The oldest and most instructive algorithm in the field. Keep a table $Q[s, a]$, act epsilon-greedily, and after each transition nudge the entry towards the bootstrapped target:

$$Q(s, a) \leftarrow Q(s, a) + \alpha\big[\underbrace{r + \gamma \max_{a'} Q(s', a') - Q(s, a)}_{\text{TD error}}\big]$$

Two properties visible in the code below:

- **Off-policy.** The `max` over $a'$ evaluates the *greedy* policy, while the agent behaves epsilon-greedily. Learning about one policy while following another is exactly what makes replay buffers and offline RL possible at all.
- **Bootstrapping.** The target contains $Q$ itself, so early updates are built on nonsense and the estimates converge from the terminal states backwards. Watch the learning curve stay flat and then take off - that is reward information propagating back from the goal.

`epsilon` decays from 1.0 to 0.05: explore first, exploit later. This is the crudest possible exploration strategy and it is enough here; in a large state space it is hopeless, which is what count-based bonuses, curiosity and Thompson sampling exist to fix.

Its limits are visible in the printed policy below. The arrows in the top-right corner - states the agent stops visiting once it has found a good route - are stale, because Q-learning only updates entries it actually reaches. The policy is optimal along the path it uses and arbitrary everywhere else, which is exactly the failure that matters when the environment changes and the agent is suddenly somewhere it stopped exploring.

------------------------------------------------------------------------

``` python
def q_learning(environment, episodes=3000, alpha=0.1, gamma=0.99,
               eps_start=1.0, eps_end=0.05, max_steps=200, seed=0):
    "Tabular Q-learning with epsilon-greedy exploration. Returns (Q, per-episode returns)."
    rng = np.random.default_rng(seed)
    Q = np.zeros((environment.n_states, environment.n_actions), dtype="float32")
    curve, td_errors = [], []

    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * np.exp(-3.0 * ep / episodes)
        s, total, done, steps = environment.reset(), 0.0, False, 0
        while not done and steps < max_steps:
            a = int(rng.integers(0, environment.n_actions)) if rng.random() < eps \
                else int(Q[s].argmax())
            s2, r, done = environment.step(a)
            # The bootstrap target: 0 beyond a terminal state, else the greedy value.
            target = r + (0.0 if done else gamma * Q[s2].max())
            td = target - Q[s, a]
            Q[s, a] += alpha * td
            td_errors.append(abs(float(td)))
            s, total, steps = s2, total + r, steps + 1
        curve.append(total)
    return Q, np.array(curve), np.array(td_errors)


t0 = time.perf_counter()
Q, q_curve, q_td = q_learning(env)
print(f"3,000 episodes in {time.perf_counter() - t0:.1f}s")

greedy_returns = rollout(lambda s: int(Q[s].argmax()), env)
print(f"\nlearned greedy policy: mean return {greedy_returns.mean():7.1f} "
      f"+/- {greedy_returns.std():.1f}   "
      f"reached the goal in {(greedy_returns > 0).mean():.1%} of episodes")
print(f"random policy:         mean return {random_returns.mean():7.1f}")
print(f"\nmean |TD error| fell from {q_td[:2000].mean():.2f} (first 2k updates) "
      f"to {q_td[-2000:].mean():.2f} (last 2k) - the value estimates stopped moving")
```

    3,000 episodes in 1.0s

    learned greedy policy: mean return    79.1 +/- 9.7   reached the goal in 99.5% of episodes
    random policy:         mean return   -94.0

    mean |TD error| fell from 1.55 (first 2k updates) to 0.72 (last 2k) - the value estimates stopped moving

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import HeatMap, Line


def smooth(x, k=50):
    "Moving average, for learning curves that are otherwise unreadable."
    return np.convolve(x, np.ones(k) / k, mode="valid")


q_line = (
    Line()
    .add_xaxis([str(i) for i in range(len(smooth(q_curve)))])
    .add_yaxis("Q-learning (50-episode mean)", [round(float(v), 2) for v in smooth(q_curve)],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("random policy", [round(float(random_returns.mean()), 2)] * len(smooth(q_curve)),
               is_smooth=False, symbol="none",
               linestyle_opts=opts.LineStyleOpts(type_="dashed"),
               label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Tabular Q-learning on the windy gridworld",
            subtitle="flat, then a sharp climb: reward information propagating back from the goal"),
        xaxis_opts=opts.AxisOpts(name="episode", axislabel_opts=opts.LabelOpts(interval=499)),
        yaxis_opts=opts.AxisOpts(name="episode return"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
q_line.render_notebook()
```

``` python
# The learned state values V(s) = max_a Q(s, a), as a heatmap over the grid. The
# gradient towards the goal IS the solution - a policy is just "walk uphill".
V = Q.max(axis=1).reshape(env.size, env.size)
cells = [[c, env.size - 1 - r, round(float(V[r, c]), 1)]
         for r in range(env.size) for c in range(env.size)]

heat = (
    HeatMap()
    .add_xaxis([str(c) for c in range(env.size)])
    .add_yaxis("V(s)", [str(env.size - 1 - r) for r in range(env.size)], cells,
               label_opts=opts.LabelOpts(is_show=True, position="inside", font_size=9))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Learned state values V(s) = max_a Q(s, a)",
            subtitle="goal at bottom-right (row 9, col 9); the dark band in row 8 is the lava"),
        visualmap_opts=opts.VisualMapOpts(min_=float(V.min()), max_=float(V.max()),
                                          is_calculable=True, orient="vertical", pos_left="right"),
        xaxis_opts=opts.AxisOpts(name="column"),
        yaxis_opts=opts.AxisOpts(name="row (0 = top)"),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
)
print("greedy action per cell (rows top to bottom):")
arrows = {0: "^", 1: "v", 2: "<", 3: ">"}
for r in range(env.size):
    row = []
    for c in range(env.size):
        if (r, c) in env.lava:
            row.append("#")
        elif (r, c) == env.goal:
            row.append("G")
        else:
            row.append(arrows[int(Q[r * env.size + c].argmax())])
    print("  " + " ".join(row))
heat.render_notebook()
```

    greedy action per cell (rows top to bottom):
      v v v < ^ ^ ^ v > <
      v v < # # # # > v ^
      v v v v v v v < v v
      > > > v v v v v v v
      > > v v > v v < v v
      ^ > > v v v > v v v
      > > v > > v > > v v
      > > > > > v v > v v
      > > > > > v > v v v
      < > > > > > > > > G

## 9. REINFORCE: the policy gradient in its rawest form

Instead of learning values and acting greedily, parameterise the policy itself and follow the gradient of expected return:

$$\nabla_\theta J(\theta) = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t\big]$$

In code that is: run an episode, compute the return from each step onward, and do a gradient step on `-(log_prob * return).sum()`. The intuition is “make the actions in good episodes more likely”, and it is that literal.

**It works, and it is terrible.** The estimator is unbiased but its variance scales with the episode length and the reward magnitude: every action in a successful episode gets credited, including the bad ones, and the algorithm has to average that out over many episodes. Two standard fixes appear below and in section 10:

- **Return normalisation** (subtract the batch mean, divide by the standard deviation) - a variance reduction that costs one line and is not optional in practice.
- **A learned baseline** - subtract $V(s)$ to get the advantage. That is actor-critic, section 10.

The policy here is a one-hidden-layer MLP over a one-hot state, which is a deliberately over-engineered way to represent 100 states - the point is that the *algorithm* is identical when the state is an image.

------------------------------------------------------------------------

``` python
class PolicyNet(nn.Module):
    "Categorical policy over the 4 actions, from a one-hot state."

    def __init__(self, n_states, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_states, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, s_onehot):
        return self.net(s_onehot)


def one_hot(states, n):
    x = torch.zeros(len(states), n, device=device)
    x[torch.arange(len(states)), torch.tensor(states, device=device)] = 1.0
    return x


def discounted_returns(rewards, gamma=0.99):
    "G_t = r_t + gamma * G_{t+1}, computed backwards."
    out, running = np.zeros(len(rewards), dtype="float32"), 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def run_episode(policy, environment, max_steps=200):
    "Sample one episode under the current policy; return the transition lists."
    states, actions, rewards = [], [], []
    s, done, steps = environment.reset(), False, 0
    while not done and steps < max_steps:
        with torch.inference_mode():
            probs = torch.softmax(policy(one_hot([s], environment.n_states))[0], -1)
        a = int(torch.multinomial(probs, 1).item())
        s2, r, done = environment.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s, steps = s2, steps + 1
    return states, actions, rewards


def reinforce(environment, episodes=6000, batch=10, lr=3e-3, gamma=0.99, seed=0):
    "REINFORCE with return normalisation. Returns (policy, curve, per-batch gradient norms)."
    torch.manual_seed(seed)
    policy = PolicyNet(environment.n_states, environment.n_actions).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    curve, credit_spread = [], []

    for it in range(episodes // batch):
        all_s, all_a, all_g = [], [], []
        for _ in range(batch):
            s, a, r = run_episode(policy, environment)
            all_s += s
            all_a += a
            all_g.append(discounted_returns(r, gamma))
            curve.append(float(np.sum(r)))
        g = np.concatenate(all_g)
        # Normalising the batch of returns IS a baseline - a constant one. It costs a
        # line and REINFORCE does not converge here without it. Section 10 replaces the
        # constant with a state-dependent baseline, which is the real fix.
        g = (g - g.mean()) / (g.std() + 1e-8)
        credit_spread.append(float(g.std()))          # the weight each action is scaled by

        logits = policy(one_hot(all_s, environment.n_states))
        log_probs = torch.log_softmax(logits, -1)[
            torch.arange(len(all_a)), torch.tensor(all_a, device=device)]
        loss = -(log_probs * torch.tensor(g, device=device)).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
    return policy, np.array(curve), np.array(credit_spread)


t0 = time.perf_counter()
pg_policy, pg_curve, pg_return_spread = reinforce(env)
print(f"6,000 episodes in {time.perf_counter() - t0:.0f}s")

pg_returns = rollout(
    lambda s: int(torch.softmax(pg_policy(one_hot([s], env.n_states))[0], -1).argmax()), env)
print(f"\nREINFORCE greedy policy: mean return {pg_returns.mean():7.1f} "
      f"+/- {pg_returns.std():.1f}   goal reached {(pg_returns > 0).mean():.1%}")
vram("after REINFORCE")
```

    6,000 episodes in 29s

    REINFORCE greedy policy: mean return    79.7 +/- 2.8   goal reached 100.0%
    VRAM after REINFORCE     0.02 GB allocated /  0.03 GB reserved

## 10. Actor-Critic: the same gradient, far less variance

REINFORCE credits every action in an episode with the whole return that followed. Actor-critic subtracts a learned estimate of how good the state already was:

$$A(s_t, a_t) \approx r_t + \gamma V(s_{t+1}) - V(s_t)$$

This is the **advantage** - “how much better than expected was this action” - and swapping $G_t$ for $A_t$ leaves the expected gradient unchanged while removing most of its variance. Subtracting any function of the state alone is provably unbiased, which is the theorem that licenses the whole family.

The **critic** $V_\phi(s)$ is trained by regression on the same bootstrapped target the Bellman equation gives, so the network has two heads and one forward pass serves both.

One implementation detail in the cell is not cosmetic. The critic regresses on the **normalised** return, not the raw one. Left in raw units - which run from about -200 to +100 in this environment - the value loss is orders of magnitude larger than the actor loss, its gradients dominate the shared trunk, and the policy never learns at all. Measured on this box: 11% goal rate with the raw target, 99% with the normalised one. Loss scaling between heads is the most common way an actor-critic implementation silently fails.

The comparison the cell makes is deliberately **within one run**: on each batch it records the spread of the return the actor would have been scaled by, and the spread of the advantage it is actually scaled by. Both come from the same data and the same policy, so the difference between them is the baseline’s contribution and nothing else.

------------------------------------------------------------------------

``` python
class ActorCritic(nn.Module):
    "Shared trunk, two heads: pi(a|s) and V(s)."

    def __init__(self, n_states, n_actions, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_states, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, s_onehot):
        h = self.trunk(s_onehot)
        return self.actor(h), self.critic(h).squeeze(-1)


def actor_critic(environment, episodes=6000, batch=10, lr=3e-3, gamma=0.99,
                 value_coef=0.5, entropy_coef=0.01, seed=0):
    "Batched advantage actor-critic (A2C) with an entropy bonus."
    torch.manual_seed(seed)
    model = ActorCritic(environment.n_states, environment.n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve, return_spread, advantage_spread = [], [], []

    for it in range(episodes // batch):
        all_s, all_a, all_g = [], [], []
        for _ in range(batch):
            s_list, a_list, r_list = [], [], []
            s, done, steps = environment.reset(), False, 0
            while not done and steps < 200:
                with torch.inference_mode():
                    logits, _ = model(one_hot([s], environment.n_states))
                a = int(torch.multinomial(torch.softmax(logits[0], -1), 1).item())
                s2, r, done = environment.step(a)
                s_list.append(s)
                a_list.append(a)
                r_list.append(r)
                s, steps = s2, steps + 1
            all_s += s_list
            all_a += a_list
            all_g.append(discounted_returns(r_list, gamma))
            curve.append(float(np.sum(r_list)))

        g_raw = torch.tensor(np.concatenate(all_g), device=device)
        # Normalise the return BEFORE it becomes the critic's regression target. Left in
        # raw units (which run from about -200 to +100 here) the MSE term is orders of
        # magnitude larger than the actor's, its gradients dominate, and the policy never
        # learns - measured on this box: 11% goal rate instead of 99%.
        g = (g_raw - g_raw.mean()) / (g_raw.std() + 1e-8)

        logits, values = model(one_hot(all_s, environment.n_states))

        # The advantage. detach() matters: the actor must not push the critic around.
        advantage = (g - values).detach()

        # The controlled measurement for section 11: on this same batch, how spread out
        # is the raw return the actor would have been scaled by, against the advantage
        # it is actually scaled by? That difference IS what the learned baseline buys.
        return_spread.append(float(g.std()))
        advantage_spread.append(float(advantage.std()))

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        log_probs_all = torch.log_softmax(logits, -1)
        log_probs = log_probs_all[torch.arange(len(all_a)), torch.tensor(all_a, device=device)]
        entropy = -(log_probs_all.exp() * log_probs_all).sum(-1).mean()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, g)               # regression on the scaled return
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    return model, np.array(curve), np.array(return_spread), np.array(advantage_spread)


t0 = time.perf_counter()
ac_model, ac_curve, ac_ret_spread, ac_adv_spread = actor_critic(env)
print(f"6,000 episodes in {time.perf_counter() - t0:.0f}s")

ac_returns = rollout(
    lambda s: int(torch.softmax(ac_model(one_hot([s], env.n_states))[0][0], -1).argmax()), env)
print(f"\nactor-critic greedy policy: mean return {ac_returns.mean():7.1f} "
      f"+/- {ac_returns.std():.1f}   goal reached {(ac_returns > 0).mean():.1%}")

print("\nspread of the credit signal, measured on the same batches:")
for label, window in (("first 50 updates", slice(0, 50)),
                      ("middle", slice(len(ac_ret_spread) // 2 - 25,
                                       len(ac_ret_spread) // 2 + 25)),
                      ("last 50 updates", slice(-50, None))):
    r, a = ac_ret_spread[window].mean(), ac_adv_spread[window].mean()
    print(f"  {label:17s} std(return) {r:.3f} -> std(return - V(s)) {a:.3f}"
          f"   ({1 - a / r:.0%} smaller)")
print("\nThe reduction starts near zero because the critic is untrained, and grows as V(s) "
      "learns.\nThat is the whole contribution of the critic, isolated: same expected "
      "gradient, less noise.")
memory_report("after actor-critic")
```

    6,000 episodes in 29s

    actor-critic greedy policy: mean return    80.3 +/- 2.1   goal reached 100.0%

    spread of the credit signal, measured on the same batches:
      first 50 updates  std(return) 1.000 -> std(return - V(s)) 0.976   (2% smaller)
      middle            std(return) 1.000 -> std(return - V(s)) 0.326   (67% smaller)
      last 50 updates   std(return) 1.000 -> std(return - V(s)) 0.351   (65% smaller)

    The reduction starts near zero because the critic is untrained, and grows as V(s) learns.
    That is the whole contribution of the critic, isolated: same expected gradient, less noise.
    RAM  after actor-critic 15.79 / 20.97 GB
    VRAM after actor-critic  0.02 GB allocated /  0.03 GB reserved

## 11. Head-to-head Benchmark

Three algorithms, the same environment, the same episode budget, and the same evaluation protocol: 200 greedy rollouts after training, plus the random-policy floor.

Read it with the section 4 caveats in mind: these are single-seed runs on a 100-state gridworld, so the ranking is indicative and the *shapes* of the learning curves carry more information than the final numbers.

The result worth noticing is that all three converge to essentially the same policy, and the cost of getting there differs by a factor of thirty. Tabular Q-learning solves this in **under a second** on 3,000 episodes; the two policy-gradient methods need **about half a minute** and twice the episodes to reach the same place. That ordering is not a quirk of the implementation - on a small discrete state space a table is simply the right representation, and the function-approximation machinery is pure overhead. It inverts completely once the state is an image, which is the only reason anyone uses policy gradients.

Hardware: knowledge-lab, RTX 3060 and 4 vCPU. The policy-gradient runs are dominated by the Python environment loop, not the network - 6,000 sequential episodes with a forward pass per step.

------------------------------------------------------------------------

``` python
import pandas as pd

EVAL = {
    "random": random_returns,
    "Q-learning (tabular)": greedy_returns,
    "REINFORCE": pg_returns,
    "actor-critic (A2C)": ac_returns,
}

rows = []
for name, ret in EVAL.items():
    rows.append(dict(algorithm=name,
                     mean_return=float(ret.mean()),
                     iqm_return=iqm(ret),
                     std=float(ret.std()),
                     goal_rate=float((ret > 0).mean()),
                     ci_low=bootstrap_ci(ret)[0], ci_high=bootstrap_ci(ret)[1]))

bench = pd.DataFrame(rows).sort_values("mean_return", ascending=False).reset_index(drop=True)
memory_report("after benchmark")
bench.round(3)
```

    RAM  after benchmark    15.83 / 20.97 GB
    VRAM after benchmark     0.02 GB allocated /  0.03 GB reserved

|  | algorithm | mean_return | iqm_return | std | goal_rate | ci_low | ci_high |
|----|----|----|----|----|----|----|----|
| 0 | actor-critic (A2C) | 80.255 | 80.434 | 2.095 | 1.000 | 79.96 | 80.540 |
| 1 | REINFORCE | 79.735 | 80.271 | 2.847 | 1.000 | 79.33 | 80.130 |
| 2 | Q-learning (tabular) | 79.120 | 80.054 | 9.723 | 0.995 | 77.57 | 80.045 |
| 3 | random | -94.010 | -79.667 | 51.727 | 0.025 | -101.26 | -86.870 |

``` python
# The runs are different lengths (3,000 vs 6,000 episodes), so the x axis is the
# fraction of each run completed - which is the honest way to compare their shapes.
curve_line = Line().add_xaxis([f"{p:.0%}" for p in np.linspace(0, 1, 200)])
for name, curve in [("Q-learning (3,000 ep)", q_curve), ("REINFORCE (6,000 ep)", pg_curve),
                    ("actor-critic (6,000 ep)", ac_curve)]:
    sm = smooth(curve)
    resampled = np.interp(np.linspace(0, len(sm) - 1, 200), np.arange(len(sm)), sm)
    curve_line.add_yaxis(name, [round(float(v), 2) for v in resampled], is_smooth=True,
                         symbol="none", label_opts=opts.LabelOpts(is_show=False))
curve_line.set_global_opts(
    title_opts=opts.TitleOpts(title="Learning curves, normalised to run length",
                              subtitle="50-episode moving average - the shape matters more than the endpoint"),
    xaxis_opts=opts.AxisOpts(name="fraction of the run",
                             axislabel_opts=opts.LabelOpts(interval=19)),
    yaxis_opts=opts.AxisOpts(name="episode return"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
curve_line.render_notebook()
```

``` python
from pyecharts.charts import Bar

# Both series come from the SAME actor-critic run and the same batches, so the gap is
# the baseline's effect and nothing else.
spread_line = (
    Line()
    .add_xaxis([str(i) for i in range(len(smooth(ac_ret_spread, 20)))])
    .add_yaxis("std of the return (what REINFORCE scales by)",
               [round(float(v), 3) for v in smooth(ac_ret_spread, 20)],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("std of the advantage (what A2C scales by)",
               [round(float(v), 3) for v in smooth(ac_adv_spread, 20)],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="What the critic actually buys",
            subtitle="subtracting a learned V(s) shrinks the credit signal's spread as the critic learns"),
        xaxis_opts=opts.AxisOpts(name="update", axislabel_opts=opts.LabelOpts(interval=99)),
        yaxis_opts=opts.AxisOpts(name="std of the per-action weight", min_=0),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
del pg_policy, ac_model
free_memory()
vram("after freeing policies")
spread_line.render_notebook()
```

    VRAM after freeing policies  0.02 GB allocated /  0.02 GB reserved

## 12. Decision Transformer: offline RL as sequence modelling

Everything above needed an environment to interact with. Most real problems do not offer one - you have a log of what happened and no ability to experiment. That is **offline RL**, and its central difficulty is distribution shift: a Q-function asked about actions the data never contains extrapolates confidently and wrongly, and the policy then optimises that fantasy.

**Decision Transformer** (Chen et al., 2021) sidesteps the problem by refusing to do RL at all. A trajectory is flattened into a token sequence

$$(\hat R_1, s_1, a_1,\ \hat R_2, s_2, a_2,\ \dots)$$

where $\hat R_t = \sum_{t' \ge t} r_{t'}$ is the **return-to-go**, and a GPT-style causal transformer is trained to predict the next action. No Bellman backup, no bootstrapping, no critic - just supervised next-token prediction on trajectories.

The interesting part is inference: **you specify the return you want**. Prime the sequence with a high return-to-go and the model generates the actions that, in its training data, preceded that outcome. It is conditional generation, and it turns “learn a good policy” into “condition on being good”.

The cell runs a real checkpoint (`edbeeching/decision-transformer-gym-hopper-medium`, 0.86M parameters) against real D4RL data - 2,186 logged Hopper trajectories. There is no MuJoCo installed here, so it is evaluated **open-loop**: feed the model the recorded states and returns-to-go, and compare its action predictions to what the behaviour policy actually did. That measures whether the model has learned the mapping, not whether it can control the robot; the closed-loop version needs `gymnasium[mujoco]` and is a few lines more (noted at the end of the cell).

Two mechanics that are easy to get wrong and are handled below:

- **States must be normalised** with the dataset’s own mean and standard deviation, and **returns-to-go scaled** (by 1000 for Hopper). The model was trained that way; feeding raw units produces confident nonsense.
- **Context is a sliding window of K=20 timesteps**, and the model reads `timesteps` as a positional input, so the absolute episode step matters, not just the position in the window.

------------------------------------------------------------------------

``` python
from huggingface_hub import hf_hub_download
from transformers import DecisionTransformerModel

DT_ID = "edbeeching/decision-transformer-gym-hopper-medium"
dt = DecisionTransformerModel.from_pretrained(DT_ID, cache_dir=HF_CACHE).to(device).eval()
print(f"{DT_ID}: {sum(p.numel() for p in dt.parameters()) / 1e6:.2f}M params, "
      f"state_dim {dt.config.state_dim}, act_dim {dt.config.act_dim}, "
      f"max_ep_len {dt.config.max_ep_len}")

# The matching D4RL trajectories (the dataset repo is script-based, so fetch the pickle).
pkl = hf_hub_download("edbeeching/decision_transformer_gym_replay",
                      "data/hopper-medium-v2.pkl", repo_type="dataset", cache_dir=HF_CACHE)
trajectories = pickle.load(open(pkl, "rb"))

ep_returns = np.array([float(t["rewards"].sum()) for t in trajectories])
ep_lengths = np.array([len(t["rewards"]) for t in trajectories])
print(f"\n{len(trajectories):,} trajectories   "
      f"return: mean {ep_returns.mean():.0f}, max {ep_returns.max():.0f}   "
      f"length: mean {ep_lengths.mean():.0f}, max {ep_lengths.max()}")
print('"medium" means the behaviour policy was mediocre on purpose - roughly a third of '
      "expert.\nThat is the point: offline RL is judged on whether it beats the data it "
      "was given.")

# Normalisation statistics from the dataset, exactly as the checkpoint expects.
all_states = np.concatenate([t["observations"] for t in trajectories])
STATE_MEAN = all_states.mean(0)
STATE_STD = all_states.std(0) + 1e-6
RETURN_SCALE = 1000.0
del all_states
free_memory()
vram("DT loaded")
```

    [transformers] Model config: bos_token_id must be `None` or an integer within the vocabulary (between 0 and 0), got 50256. This may result in unexpected behavior.
    [transformers] Model config: eos_token_id must be `None` or an integer within the vocabulary (between 0 and 0), got 50256. This may result in unexpected behavior.

    [transformers] DecisionTransformerModel LOAD REPORT from: edbeeching/decision-transformer-gym-hopper-medium
    Key                                  | Status     |  | 
    -------------------------------------+------------+--+-
    encoder.h.{0, 1, 2}.attn.masked_bias | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

    edbeeching/decision-transformer-gym-hopper-medium: 0.86M params, state_dim 11, act_dim 3, max_ep_len 1000

    2,186 trajectories   return: mean 1422, max 3222   length: mean 457, max 1000
    "medium" means the behaviour policy was mediocre on purpose - roughly a third of expert.
    That is the point: offline RL is judged on whether it beats the data it was given.
    VRAM DT loaded           0.02 GB allocated /  0.03 GB reserved

``` python
K = 20              # the context window the checkpoint was trained with


def dt_predict_actions(traj, horizon=120):
    """Open-loop: replay recorded states and returns-to-go, read off predicted actions.

    A closed-loop rollout would feed the model's own action into a simulator and take
    the next state from there. That needs gymnasium[mujoco]; this measures the same
    learned mapping without one.
    """
    n = min(horizon, len(traj["observations"]))
    states = (traj["observations"][:n] - STATE_MEAN) / STATE_STD
    actions = traj["actions"][:n]
    rtg = np.cumsum(traj["rewards"][:n][::-1])[::-1] / RETURN_SCALE     # return-to-go

    preds = np.zeros_like(actions)
    with torch.inference_mode():
        for t in range(n):
            lo = max(0, t - K + 1)
            s = torch.tensor(states[lo:t + 1], dtype=torch.float32, device=device)[None]
            a = torch.tensor(actions[lo:t + 1], dtype=torch.float32, device=device)[None]
            r = torch.tensor(rtg[lo:t + 1].copy(), dtype=torch.float32,
                             device=device)[None, :, None]
            ts = torch.arange(lo, t + 1, device=device)[None]           # absolute step index
            mask = torch.ones(1, t - lo + 1, dtype=torch.long, device=device)
            out = dt(states=s, actions=a, returns_to_go=r, timesteps=ts,
                     attention_mask=mask, return_dict=True)
            preds[t] = out.action_preds[0, -1].cpu().numpy()
    return preds, actions


# A high-return and a low-return trajectory, to see whether quality changes the fit.
best_i, worst_i = int(np.argmax(ep_returns)), int(np.argmin(ep_returns[ep_lengths > 150]))
worst_i = int(np.where(ep_lengths > 150)[0][worst_i])

t0 = time.perf_counter()
rows = []
for label, i in (("highest-return trajectory", best_i), ("low-return trajectory", worst_i)):
    pred, true = dt_predict_actions(trajectories[i])
    mae = float(np.abs(pred - true).mean())
    corr = [float(np.corrcoef(pred[:, j], true[:, j])[0, 1]) for j in range(true.shape[1])]
    rows.append(dict(trajectory=label, episode_return=float(ep_returns[i]),
                     action_mae=mae, mean_corr=float(np.mean(corr))))
    print(f"{label:26s} return {ep_returns[i]:7.1f}   action MAE {mae:.3f}   "
          f"per-joint correlation {[round(c, 2) for c in corr]}")
print(f"\n{time.perf_counter() - t0:.1f}s")

# The scale to judge MAE against: actions live in [-1, 1].
print(f"actions are bounded in [-1, 1]; the standard deviation of the recorded actions "
      f"is {trajectories[best_i]['actions'].std():.3f}")
pred_best, true_best = dt_predict_actions(trajectories[best_i])
```

    highest-return trajectory  return  3222.4   action MAE 0.183   per-joint correlation [0.89, 0.89, 0.87]
    low-return trajectory      return   331.3   action MAE 0.208   per-joint correlation [0.92, 0.89, 0.7]

    0.3s
    actions are bounded in [-1, 1]; the standard deviation of the recorded actions is 0.647

``` python
# Predicted against recorded torque for one joint. Where the two track each other, the
# model has learned the behaviour policy's mapping from (return-to-go, state) to action.
JOINT = 0
steps = list(range(len(true_best)))
act_line = (
    Line()
    .add_xaxis(steps)
    .add_yaxis("recorded action", [round(float(v), 3) for v in true_best[:, JOINT]],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("Decision Transformer prediction",
               [round(float(v), 3) for v in pred_best[:, JOINT]],
               is_smooth=True, symbol="none", label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"Hopper joint {JOINT}: predicted vs recorded torque",
            subtitle=f"open-loop on the highest-return trajectory (return {ep_returns[best_i]:.0f})"),
        xaxis_opts=opts.AxisOpts(type_="value", name="timestep"),
        yaxis_opts=opts.AxisOpts(name="action (torque, [-1, 1])"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
act_line.render_notebook()
```

``` python
# The return distribution of the offline dataset. "Medium" data is the hard case for
# offline RL: an algorithm that only imitates is capped at this distribution, and the
# whole research question is whether it can stitch together something better.
counts, edges = np.histogram(ep_returns, bins=40)
ret_hist = (
    Bar()
    .add_xaxis([f"{c:.0f}" for c in (edges[:-1] + edges[1:]) / 2])
    .add_yaxis("trajectories", [int(c) for c in counts], category_gap="0%",
               label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="hopper-medium-v2: what the offline data actually contains",
            subtitle=f"{len(trajectories):,} trajectories, mean return {ep_returns.mean():.0f}, "
                     f"best {ep_returns.max():.0f}"),
        xaxis_opts=opts.AxisOpts(name="episode return",
                                 axislabel_opts=opts.LabelOpts(rotate=45, font_size=9)),
        yaxis_opts=opts.AxisOpts(name="count"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
print("To close the loop and get a real score, install a simulator and roll out:")
print("""
  # %pip install "gymnasium[mujoco]"
  # import gymnasium as gym
  # env = gym.make("Hopper-v4")
  # obs, _ = env.reset(); target_return = 3600 / RETURN_SCALE   # ask for expert-level
  # ... at each step: append the normalised obs, run dt(...), take action_preds[0, -1],
  #     step the env, and subtract the received reward from the return-to-go.
""")
del dt
free_memory()
vram("after DT")
ret_hist.render_notebook()
```

    To close the loop and get a real score, install a simulator and roll out:

      # %pip install "gymnasium[mujoco]"
      # import gymnasium as gym
      # env = gym.make("Hopper-v4")
      # obs, _ = env.reset(); target_return = 3600 / RETURN_SCALE   # ask for expert-level
      # ... at each step: append the normalised obs, run dt(...), take action_preds[0, -1],
      #     step the env, and subtract the received reward from the return-to-go.

    VRAM after DT            0.02 GB allocated /  0.02 GB reserved

## 13. RL for Language Models: where RL actually became ubiquitous

The most consequential RL deployment in the world is not a robot or a game. It is the post-training step that turns a base language model into an assistant, and it runs on essentially every frontier model shipped since 2022.

**The RLHF recipe (InstructGPT, 2022).**

1.  **SFT** - supervised fine-tuning on demonstrations. This is not RL, and it does most of the work.

2.  **Reward model** - collect human comparisons between pairs of responses and fit a model $r_\phi(x, y)$ under the Bradley-Terry likelihood $P(y_1 \succ y_2) = \sigma(r_\phi(x, y_1) - r_\phi(x, y_2))$.

3.  **PPO against the reward model**, with a KL penalty pinning the policy near the SFT model:

    $$\max_\pi \ \mathbb{E}_{y \sim \pi}\big[r_\phi(x, y)\big] - \beta\, D_{KL}\!\big(\pi \,\|\, \pi_{\text{SFT}}\big)$$

    The KL term is the whole safety mechanism. Without it the policy walks off into whatever adversarial region of token space the reward model happens to score highly - which is reward hacking, arriving on schedule.

The MDP mapping is unusual: the **state** is the prompt plus the tokens generated so far, an **action** is one token, the episode is one response, and the reward is a single scalar at the end. It is a bandit problem wearing an MDP’s clothes.

**DPO (2023) removed the RL.** Rafailov et al. showed the RLHF objective has a closed-form optimal policy, and reparameterising the reward in terms of the policy turns the whole thing into a supervised classification loss on preference pairs:

$$\mathcal{L}_{DPO} = -\log \sigma\!\Big(\beta \log \tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
                                        - \beta \log \tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)$$

No reward model, no sampling loop, no value network. It is dramatically simpler and became the default for preference tuning almost immediately.

**GRPO and RLVR (2024-2025) brought the RL back, for a reason.** When the reward is **verifiable** - the maths answer checks out, the unit test passes, the proof compiles - you do not need a learned reward model, and the reward cannot be hacked in the usual way. **GRPO** (Group Relative Policy Optimization, DeepSeekMath 2024) also drops the value network: sample $G$ completions for the same prompt and use the group’s mean reward as the baseline,

$$A_i = \frac{r_i - \text{mean}(r_1 \dots r_G)}{\text{std}(r_1 \dots r_G)}$$

which is REINFORCE-with-a-baseline (section 10) applied at the group level, and saves the memory of a second full-size critic. **DeepSeek-R1 (2025)** showed that pure RL on verifiable rewards makes long chain-of-thought reasoning emerge without any reasoning demonstrations at all - the model learns to think longer because thinking longer gets the answer right.

**How to actually run these.** Hugging Face **`trl`** implements all three: `PPOTrainer`, `DPOTrainer`, `GRPOTrainer`, each a thin wrapper over `transformers`. A 1.5B model with LoRA and 4-bit quantisation is roughly the ceiling of what fits in 12 GB of VRAM, so it is feasible on this box - but a real GRPO run generates $G$ completions per prompt for thousands of prompts, and that is a datacentre-scale amount of inference, not a notebook cell. That is why this section is prose: the honest demonstration does not fit, and a toy one would misrepresent the cost.

------------------------------------------------------------------------

## 14. Common Frameworks

RL’s ecosystem is organised around a distinction that does not exist in the rest of this folder: the environment is part of the system. Half the frameworks below are environments or simulators rather than models, and the single most common cause of an RL result that will not reproduce is an environment version difference. The other organising fact is that RL’s largest deployment by far is post-training language models, and that has its own stack.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | modelling | Tested PPO, SAC, TD3 and DQN implementations with sane defaults and a callback system | MIT | You want a *working* implementation. PPO or SAC here beats anything you write in a week |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | modelling | Every algorithm as one self-contained file with no abstraction layers | MIT | You want to *understand* PPO, or to modify it. The fastest path from the equations to running code |
| [Ray RLlib](https://github.com/ray-project/ray) / [TorchRL](https://github.com/pytorch/rl) / [Tianshou](https://github.com/thu-ml/tianshou) | modelling | Distributed rollouts and training, and composable RL primitives when you are building something new | Apache 2.0 / MIT | Scale, or a custom algorithm. RLlib is the one that runs across a cluster |
| [trl](https://github.com/huggingface/trl) / [verl](https://github.com/volcengine/verl) / [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | modelling | RL on language models: DPO for preference pairs, GRPO/PPO for verifiable rewards, with the rollout and training loops sharded | Apache 2.0 | This is where RL is actually ubiquitous now - see section 13 |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) + [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) | data | The environment API everything above expects, plus classic control, Box2D, MuJoCo and the multi-agent equivalent | MIT | Always. Pin the environment version with the result - a reward curve without it is not reproducible |
| [MuJoCo](https://github.com/google-deepmind/mujoco) / [Isaac Lab](https://github.com/isaac-sim/IsaacLab) / [EnvPool](https://github.com/sail-sg/envpool) | data | Fast physics, GPU-parallel environments, and vectorised stepping - which is usually the throughput bottleneck, not the network | Apache 2.0 / BSD-3 | Continuous control. On 4 cores, environment stepping dominates wall clock and vectorisation is the fix |
| [Minari](https://github.com/Farama-Foundation/Minari) / D4RL | data | Standard offline datasets with fixed splits, for IQL, CQL and Decision Transformer | Apache 2.0 | Offline RL, which is what most industrial problems actually are - you cannot let a learning policy explore in production |
| [Weights & Biases](https://github.com/wandb/wandb) / [MLflow](https://github.com/mlflow/mlflow) + video logging | evaluation | Learning curves across seeds, and rendered rollouts - which show reward hacking that no scalar will | MIT / Apache 2.0 | Every run. Watch the trajectories, not just the reward curve |
| [rliable](https://github.com/google-research/rliable) | evaluation | IQM with stratified bootstrap confidence intervals - the Agarwal et al. (2021) recommendations implemented directly | Apache 2.0 | Reporting anything. RL results without error bars across multiple seeds are anecdotes |

**The 2026 default stack** is Gymnasium for the interface, CleanRL to understand the algorithm and Stable-Baselines3 to run it, a vectorised environment so stepping is not the bottleneck, W&B logging video as well as scalars, and rliable for the numbers you report. `trl` if the policy is a language model.

The common wrong turn is writing your own PPO. The published implementations differ from the paper in a dozen small ways - advantage normalisation, value clipping, orthogonal initialisation - and those details account for much of the performance. Read CleanRL, run Stable-Baselines3. The second is trusting the reward curve: write down in advance what an adversary could do to maximise your reward while defeating your intent, then inspect trajectories to see whether it did.

------------------------------------------------------------------------

## 15. Going Further

**Use PPO or SAC, not what you built here.** REINFORCE and A2C are in this notebook because they are legible. For anything real, PPO (on-policy, robust, few knobs) or SAC (off-policy, continuous, sample-efficient) are the defaults, and the gap between them and vanilla policy gradients is large.

**Take offline RL seriously if you cannot interact.** Most industrial problems are offline. IQL and CQL are the strong value-based options; Decision Transformer and its successors are the sequence-modelling option; and the D4RL/Minari benchmarks exist so you can compare them honestly.

**Report results properly.** Multiple seeds, IQM with stratified bootstrap confidence intervals, and learning curves rather than final numbers - the `rliable` library implements the Agarwal et al. (2021) recommendations directly. RL results without error bars are anecdotes.

**Design the reward like an adversary will read it, because one will.** Write down what the agent could do that maximises your reward and defeats your intent, before you train. Add a KL or trust-region constraint to whatever it started from, and inspect trajectories rather than trusting the reward curve.

**For language models, start with `trl`.** DPO if you have preference pairs, GRPO if you have a verifier. Both are a few dozen lines on top of `transformers`, and both matter far more in practice today than anything in the game-playing literature.

**Related notebooks in this repo:** `01_Robotics` (RL and imitation learning on physical systems), `Natural_Language_Processing/08_Text_Generation` (the policy that RLHF fine-tunes), and `Other/00_Graph_Machine_Learning` (graphs as the state space in combinatorial RL).

------------------------------------------------------------------------

Back to top
