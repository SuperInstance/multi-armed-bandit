//! # multi-armed-bandit
//!
//! Multi-armed bandit algorithm implementations for exploration-exploitation
//! decision making. Includes ε-greedy, UCB1, Thompson Sampling, and regret tracking.
//!
//! ## Example
//!
//! ```
//! use multi_armed_bandit::{EpsilonGreedy, BanditEnvironment, Arm, RegretTracker};
//!
//! let mut env = BanditEnvironment::new(vec![0.3, 0.7, 0.5]);
//! let mut agent = EpsilonGreedy::new(3, 0.1);
//! let mut tracker = RegretTracker::new(env.optimal_reward());
//!
//! for _ in 0..500 {
//!     let arm = agent.select_arm();
//!     let reward = env.pull(arm);
//!     agent.update(arm, reward);
//!     tracker.record(arm, reward, env.arm_means()[arm]);
//! }
//!
//! println!("Cumulative regret: {:.2}", tracker.cumulative_regret());
//! ```

/// A bandit arm with unknown reward distribution.
#[derive(Debug, Clone)]
pub struct Arm {
    /// Number of times this arm has been pulled.
    pulls: u64,
    /// Cumulative reward from this arm.
    total_reward: f64,
}

impl Arm {
    /// Create a new arm with no pulls.
    pub fn new() -> Self {
        Self { pulls: 0, total_reward: 0.0 }
    }

    /// Record a pull with the given reward.
    pub fn pull(&mut self, reward: f64) {
        self.pulls += 1;
        self.total_reward += reward;
    }

    /// Number of times this arm has been pulled.
    pub fn pulls(&self) -> u64 {
        self.pulls
    }

    /// Total accumulated reward.
    pub fn total_reward(&self) -> f64 {
        self.total_reward
    }

    /// Empirical mean reward (0.0 if never pulled).
    pub fn mean(&self) -> f64 {
        if self.pulls == 0 { 0.0 } else { self.total_reward / self.pulls as f64 }
    }
}

impl Default for Arm {
    fn default() -> Self {
        Self::new()
    }
}

/// ε-greedy bandit algorithm.
///
/// With probability ε, selects a random arm (exploration).
/// With probability 1-ε, selects the arm with highest empirical mean (exploitation).
#[derive(Debug, Clone)]
pub struct EpsilonGreedy {
    arms: Vec<Arm>,
    epsilon: f64,
    rng_state: u64,
}

impl EpsilonGreedy {
    /// Create a new ε-greedy agent with `n_arms` arms and exploration rate ε.
    pub fn new(n_arms: usize, epsilon: f64) -> Self {
        assert!(n_arms > 0, "must have at least one arm");
        assert!((0.0..=1.0).contains(&epsilon), "epsilon must be in [0, 1]");
        Self {
            arms: vec![Arm::new(); n_arms],
            epsilon,
            rng_state: 42,
        }
    }

    /// Simple deterministic PRNG (xorshift64).
    fn next_random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x & 0x7FFFFFFFFFFFFFFF) as f64 / u64::MAX as f64
    }

    /// Select an arm using the ε-greedy policy.
    pub fn select_arm(&mut self) -> usize {
        if self.next_random() < self.epsilon {
            // Explore: random arm
            (self.next_random() * self.arms.len() as f64) as usize
        } else {
            // Exploit: best empirical mean
            self.best_arm()
        }
    }

    /// Update the arm's statistics with the observed reward.
    pub fn update(&mut self, arm: usize, reward: f64) {
        self.arms[arm].pull(reward);
    }

    /// Get the arm with the highest empirical mean.
    pub fn best_arm(&self) -> usize {
        self.arms.iter().enumerate()
            .max_by(|a, b| a.1.mean().partial_cmp(&b.1.mean()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get the number of arms.
    pub fn n_arms(&self) -> usize {
        self.arms.len()
    }

    /// Get reference to an arm.
    pub fn arm(&self, index: usize) -> &Arm {
        &self.arms[index]
    }

    /// Set the exploration rate.
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon;
    }
}

/// UCB1 (Upper Confidence Bound) bandit algorithm.
///
/// Selects the arm that maximizes: mean + sqrt(2 * ln(t) / n_i)
/// where t is total pulls and n_i is pulls of arm i.
#[derive(Debug, Clone)]
pub struct UCB1 {
    arms: Vec<Arm>,
    total_pulls: u64,
}

impl UCB1 {
    /// Create a new UCB1 agent with `n_arms` arms.
    pub fn new(n_arms: usize) -> Self {
        assert!(n_arms > 0, "must have at least one arm");
        Self {
            arms: vec![Arm::new(); n_arms],
            total_pulls: 0,
        }
    }

    /// Compute the UCB1 index for a given arm.
    pub fn ucb_index(&self, arm: usize) -> f64 {
        let a = &self.arms[arm];
        if a.pulls() == 0 {
            return f64::INFINITY;
        }
        let exploration = (2.0 * (self.total_pulls as f64).ln() / a.pulls() as f64).sqrt();
        a.mean() + exploration
    }

    /// Select the arm with the highest UCB1 index.
    /// Unpulled arms get infinite priority (ensures each arm is tried at least once).
    pub fn select_arm(&mut self) -> usize {
        // First pass: pull each arm once
        for (i, arm) in self.arms.iter().enumerate() {
            if arm.pulls() == 0 {
                return i;
            }
        }
        // Then select by UCB index
        self.arms.iter().enumerate()
            .max_by(|a, b| {
                self.ucb_index(a.0).partial_cmp(&self.ucb_index(b.0)).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update arm statistics with observed reward.
    pub fn update(&mut self, arm: usize, reward: f64) {
        self.arms[arm].pull(reward);
        self.total_pulls += 1;
    }

    /// Get reference to an arm.
    pub fn arm(&self, index: usize) -> &Arm {
        &self.arms[index]
    }

    /// Total number of pulls across all arms.
    pub fn total_pulls(&self) -> u64 {
        self.total_pulls
    }
}

/// Thompson Sampling bandit algorithm using Beta distributions.
///
/// Maintains Beta(α, β) posterior for each arm. On each step, samples
/// from each arm's posterior and selects the arm with the highest sample.
#[derive(Debug, Clone)]
pub struct ThompsonSampling {
    /// Alpha parameters (successes + 1) for each arm.
    alphas: Vec<f64>,
    /// Beta parameters (failures + 1) for each arm.
    betas: Vec<f64>,
    rng_state: u64,
}

impl ThompsonSampling {
    /// Create a new Thompson Sampling agent with `n_arms` arms.
    /// Each arm starts with Beta(1, 1) = Uniform prior.
    pub fn new(n_arms: usize) -> Self {
        assert!(n_arms > 0, "must have at least one arm");
        Self {
            alphas: vec![1.0; n_arms],
            betas: vec![1.0; n_arms],
            rng_state: 12345,
        }
    }

    fn next_random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x & 0x7FFFFFFFFFFFFFFF) as f64 / u64::MAX as f64
    }

    /// Sample from a Beta distribution using a simple acceptance-rejection method.
    fn sample_beta(&mut self, alpha: f64, beta: f64) -> f64 {
        // Use Johnk's algorithm for small parameters
        if alpha < 1.0 && beta < 1.0 {
            loop {
                let u = self.next_random().powf(1.0 / alpha);
                let v = self.next_random().powf(1.0 / beta);
                if u + v <= 1.0 {
                    return if u + v > 0.0 { u / (u + v) } else { 0.5 };
                }
            }
        }
        // For alpha >= 1, beta >= 1: use ratio of gammas approximation
        let g1 = self.sample_gamma(alpha);
        let g2 = self.sample_gamma(beta);
        if g1 + g2 > 0.0 { g1 / (g1 + g2) } else { 0.5 }
    }

    /// Approximate gamma sample using Marsaglia and Tsang's method.
    fn sample_gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            return self.sample_gamma(shape + 1.0) * self.next_random().powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = (9.0 * d).sqrt().recip();
        loop {
            let mut x: f64;
            let mut v: f64;
            loop {
                x = self.next_standard_normal();
                v = 1.0 + c * x;
                if v > 0.0 { break; }
            }
            let v = v * v * v;
            let u = self.next_random();
            if u < 1.0 - 0.0331 * (x * x) * (x * x) {
                return d * v;
            }
            if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Standard normal sample using Box-Muller.
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_random().max(1e-15);
        let u2 = self.next_random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Select an arm by Thompson Sampling.
    pub fn select_arm(&mut self) -> usize {
        let mut best = 0;
        let mut best_sample = self.sample_beta(self.alphas[0], self.betas[0]);
        for i in 1..self.alphas.len() {
            let sample = self.sample_beta(self.alphas[i], self.betas[i]);
            if sample > best_sample {
                best_sample = sample;
                best = i;
            }
        }
        best
    }

    /// Update arm posterior with observed reward (0 or 1).
    pub fn update(&mut self, arm: usize, reward: f64) {
        if reward > 0.5 {
            self.alphas[arm] += 1.0;
        } else {
            self.betas[arm] += 1.0;
        }
    }

    /// Get the alpha parameter for an arm.
    pub fn alpha(&self, arm: usize) -> f64 {
        self.alphas[arm]
    }

    /// Get the beta parameter for an arm.
    pub fn beta(&self, arm: usize) -> f64 {
        self.betas[arm]
    }
}

/// Simulated bandit environment with configurable reward distributions.
#[derive(Debug, Clone)]
pub struct BanditEnvironment {
    /// Mean reward for each arm (Bernoulli with given probability).
    arm_means: Vec<f64>,
    rng_state: u64,
}

impl BanditEnvironment {
    /// Create an environment with given arm reward probabilities.
    pub fn new(arm_means: Vec<f64>) -> Self {
        assert!(!arm_means.is_empty(), "must have at least one arm");
        for (i, &m) in arm_means.iter().enumerate() {
            assert!((0.0..=1.0).contains(&m), "arm {} mean must be in [0,1]", i);
        }
        Self { arm_means, rng_state: 98765 }
    }

    fn next_random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x & 0x7FFFFFFFFFFFFFFF) as f64 / u64::MAX as f64
    }

    /// Pull an arm and receive a Bernoulli reward (0.0 or 1.0).
    pub fn pull(&mut self, arm: usize) -> f64 {
        if self.next_random() < self.arm_means[arm] { 1.0 } else { 0.0 }
    }

    /// Get the reward probabilities of all arms.
    pub fn arm_means(&self) -> &[f64] {
        &self.arm_means
    }

    /// Get the optimal (highest) reward probability.
    pub fn optimal_reward(&self) -> f64 {
        self.arm_means.iter().cloned().fold(0.0f64, f64::max)
    }

    /// Get the index of the best arm.
    pub fn best_arm(&self) -> usize {
        self.arm_means.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Number of arms.
    pub fn n_arms(&self) -> usize {
        self.arm_means.len()
    }
}

/// Tracks cumulative regret for evaluating bandit algorithms.
#[derive(Debug, Clone)]
pub struct RegretTracker {
    /// Cumulative regret (difference between optimal and actual reward).
    cumulative_regret: f64,
    /// Number of steps tracked.
    steps: u64,
    /// Optimal reward per step.
    optimal_reward: f64,
}

impl RegretTracker {
    /// Create a new regret tracker with the given optimal per-step reward.
    pub fn new(optimal_reward: f64) -> Self {
        Self {
            cumulative_regret: 0.0,
            steps: 0,
            optimal_reward,
        }
    }

    /// Record a step: the reward received and the mean of the chosen arm.
    pub fn record(&mut self, _arm: usize, _reward: f64, arm_mean: f64) {
        let regret = self.optimal_reward - arm_mean;
        self.cumulative_regret += regret;
        self.steps += 1;
    }

    /// Cumulative regret so far.
    pub fn cumulative_regret(&self) -> f64 {
        self.cumulative_regret
    }

    /// Average regret per step.
    pub fn average_regret(&self) -> f64 {
        if self.steps == 0 { 0.0 } else { self.cumulative_regret / self.steps as f64 }
    }

    /// Number of steps recorded.
    pub fn steps(&self) -> u64 {
        self.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm_creation() {
        let arm = Arm::new();
        assert_eq!(arm.pulls(), 0);
        assert_eq!(arm.mean(), 0.0);
    }

    #[test]
    fn test_arm_pull() {
        let mut arm = Arm::new();
        arm.pull(1.0);
        arm.pull(0.0);
        arm.pull(1.0);
        assert_eq!(arm.pulls(), 3);
        assert!((arm.mean() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_greedy_creation() {
        let agent = EpsilonGreedy::new(5, 0.1);
        assert_eq!(agent.n_arms(), 5);
    }

    #[test]
    fn test_epsilon_greedy_converges() {
        let mut agent = EpsilonGreedy::new(3, 0.0); // Pure exploitation
        // Manually update to make arm 1 clearly best
        for _ in 0..10 { agent.update(0, 0.0); }
        for _ in 0..10 { agent.update(1, 1.0); }
        for _ in 0..10 { agent.update(2, 0.0); }
        // With epsilon=0, should always pick arm 1
        for _ in 0..20 {
            assert_eq!(agent.select_arm(), 1);
        }
    }

    #[test]
    fn test_ucb1_initial_exploration() {
        let mut agent = UCB1::new(3);
        // First 3 selections should be each arm once
        let mut seen = [false; 3];
        for _ in 0..3 {
            let arm = agent.select_arm();
            seen[arm] = true;
            agent.update(arm, 1.0);
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn test_ucb1_index() {
        let mut agent = UCB1::new(2);
        agent.update(0, 1.0);
        agent.update(0, 1.0);
        agent.update(1, 0.0);
        // Unpulled arms have infinite index
        assert!(agent.ucb_index(0).is_finite());
    }

    #[test]
    fn test_thompson_sampling_creation() {
        let agent = ThompsonSampling::new(3);
        assert_eq!(agent.alpha(0), 1.0);
        assert_eq!(agent.beta(0), 1.0);
    }

    #[test]
    fn test_thompson_sampling_update() {
        let mut agent = ThompsonSampling::new(2);
        agent.update(0, 1.0); // Success
        agent.update(0, 0.0); // Failure
        assert_eq!(agent.alpha(0), 2.0);
        assert_eq!(agent.beta(0), 2.0);
    }

    #[test]
    fn test_thompson_sampling_selects() {
        let mut agent = ThompsonSampling::new(3);
        // Should always return a valid arm index
        for _ in 0..100 {
            let arm = agent.select_arm();
            assert!(arm < 3);
        }
    }

    #[test]
    fn test_bandit_environment() {
        let env = BanditEnvironment::new(vec![0.3, 0.7, 0.5]);
        assert_eq!(env.n_arms(), 3);
        assert!((env.optimal_reward() - 0.7).abs() < 1e-10);
        assert_eq!(env.best_arm(), 1);
    }

    #[test]
    fn test_regret_tracker() {
        let mut tracker = RegretTracker::new(1.0);
        tracker.record(0, 0.0, 0.5);
        tracker.record(1, 1.0, 1.0);
        assert!((tracker.cumulative_regret() - 0.5).abs() < 1e-10);
        assert_eq!(tracker.steps(), 2);
    }

    #[test]
    fn test_regret_average() {
        let mut tracker = RegretTracker::new(1.0);
        tracker.record(0, 0.0, 0.5);
        tracker.record(0, 0.0, 0.5);
        assert!((tracker.average_regret() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_greedy_best_arm() {
        let mut agent = EpsilonGreedy::new(3, 0.1);
        agent.update(0, 0.2);
        agent.update(1, 0.8);
        agent.update(2, 0.5);
        assert_eq!(agent.best_arm(), 1);
    }

    #[test]
    fn test_bandit_pulls_valid() {
        let mut env = BanditEnvironment::new(vec![0.0, 1.0]);
        // Arm 0 always returns 0, arm 1 always returns 1
        for _ in 0..10 {
            assert_eq!(env.pull(0), 0.0);
            assert_eq!(env.pull(1), 1.0);
        }
    }

    #[test]
    fn test_ucb1_convergence() {
        let mut agent = UCB1::new(3);
        // Feed rewards where arm 1 is best
        for _ in 0..200 {
            let arm = agent.select_arm();
            let reward = if arm == 1 { 1.0 } else { 0.0 };
            agent.update(arm, reward);
        }
        // Arm 1 should have the highest mean
        assert!(agent.arm(1).mean() > agent.arm(0).mean());
        assert!(agent.arm(1).mean() > agent.arm(2).mean());
    }
}
