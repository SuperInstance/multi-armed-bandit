# multi-armed-bandit

A Rust library implementing **multi-armed bandit algorithms** for exploration-exploitation decision making under uncertainty.

[![crates.io](https://img.shields.io/crates/v/multi-armed-bandit.svg)](https://crates.io/crates/multi-armed-bandit)
[![Documentation](https://docs.rs/multi-armed-bandit/badge.svg)](https://docs.rs/multi-armed-bandit)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

The multi-armed bandit problem is a classic reinforcement learning scenario: you have multiple actions (arms), each with an unknown reward distribution, and you must balance **exploration** (trying new arms) against **exploitation** (using the best arm found so far) to maximize cumulative reward.

This library provides:

- **ε-Greedy** — Simple random exploration with configurable ε
- **UCB1** — Upper Confidence Bound with optimism in the face of uncertainty
- **Thompson Sampling** — Bayesian posterior sampling for efficient exploration
- **Bandit Environment** — Simulated testbed for algorithm evaluation
- **Regret Tracker** — Measure how your algorithm compares to the optimal strategy

## Installation

```toml
[dependencies]
multi-armed-bandit = "0.1.0"
```

## Quick Start

```rust
use multi_armed_bandit::{UCB1, BanditEnvironment, RegretTracker};

// Create environment with 3 arms (reward probabilities: 0.3, 0.7, 0.5)
let mut env = BanditEnvironment::new(vec![0.3, 0.7, 0.5]);
let mut agent = UCB1::new(3);
let mut tracker = RegretTracker::new(env.optimal_reward());

// Run for 1000 steps
for _ in 0..1000 {
    let arm = agent.select_arm();
    let reward = env.pull(arm);
    agent.update(arm, reward);
    tracker.record(arm, reward, env.arm_means()[arm]);
}

println!("Best arm found: {}", /* identify from agent */);
println!("Cumulative regret: {:.2}", tracker.cumulative_regret());
println!("Average regret: {:.4}", tracker.average_regret());
```

## Algorithms

### ε-Greedy

The simplest bandit algorithm. With probability ε, select a random arm. Otherwise, select the arm with the highest empirical mean reward.

```rust
use multi_armed_bandit::EpsilonGreedy;

let mut agent = EpsilonGreedy::new(5, 0.1); // 5 arms, 10% exploration
for _ in 0..500 {
    let arm = agent.select_arm();
    let reward = /* your reward logic */;
    agent.update(arm, reward);
}
```

**Pros:** Simple, easy to tune
**Cons:** Explores uniformly, doesn't focus exploration where it's most needed

### UCB1

Upper Confidence Bound selects arms based on an optimism principle. Each arm's score includes an exploration bonus that decreases as the arm is pulled more:

```
score(i) = mean(i) + sqrt(2 * ln(t) / n_i)
```

**Pros:** Strong theoretical guarantees (O(log n) regret), no hyperparameters
**Cons:** Can be sensitive to reward scale

### Thompson Sampling

A Bayesian approach that maintains a Beta posterior for each arm and samples from these posteriors to select actions. Arms with uncertain (high-variance) posteriors are naturally explored more.

```rust
use multi_armed_bandit::ThompsonSampling;

let mut agent = ThompsonSampling::new(3);
for _ in 0..1000 {
    let arm = agent.select_arm();
    let reward = /* 0.0 or 1.0 */;
    agent.update(arm, reward);
}
```

**Pros:** Empirically excellent performance, naturally balances exploration/exploitation
**Cons:** Requires Bernoulli rewards (0/1) for Beta-Binomial model

## API Reference

| Type | Description |
|------|-------------|
| `Arm` | Tracks pulls and rewards for a single action |
| `EpsilonGreedy` | ε-greedy algorithm with configurable exploration |
| `UCB1` | Upper Confidence Bound algorithm |
| `ThompsonSampling` | Bayesian Thompson Sampling with Beta priors |
| `BanditEnvironment` | Simulated bandit with configurable reward distributions |
| `RegretTracker` | Cumulative and average regret computation |

## Performance

All algorithms run in O(k) per step where k is the number of arms. No external dependencies required. The Thompson Sampling implementation includes a Gamma distribution sampler for Beta posterior sampling.

## Applications

- **Ad placement**: Which ad gets the most clicks?
- **Clinical trials**: Which treatment is most effective?
- **A/B/n testing**: Compare multiple variants simultaneously
- **Recommendation systems**: Explore new content vs. show known favorites
- **Hyperparameter optimization**: Efficiently search configuration spaces

## Algorithm Comparison

| Algorithm | Exploration | Theoretical Regret | Best For |
|-----------|-------------|-------------------|----------|
| ε-Greedy | Random (ε) | O(√n) | Simple baselines |
| UCB1 | Optimistic bounds | O(log n) | General purpose |
| Thompson | Posterior sampling | O(log n) | Bernoulli rewards |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR at [GitHub](https://github.com/SuperInstance/multi-armed-bandit).
