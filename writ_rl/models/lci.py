from writ_tools.models.env import WordsconsinEnv
from writ_tools.models.decisionmaker import AbstractDecisionMaker
from writ_tools.models.utils import *


class SimpleCountModel(AbstractDecisionMaker):
    """
    Tracks the counts of each rule and chooses whichever has the most counts.
    In that case we can use the following model for the probability of making a choice at time $t$ ($c_t$) as:
    $$
    p(c_t = k) = \frac{N_k}{\sum_{j=1}^K N_j}, \forall k \leq K
    $$

    Where we have a separate $N_k$ for each rule, which we update on each trial according to the following rule:
    $$
    N_k \leftarrow N_k + \delta(c_t = k)
    $$
    Where $\delta(c_t = k)$ is 1 if the choice $c_t$ matches the rule $k$ and 0 otherwise.
    """

    def __init__(self, num_rules: int):
        self.num_rules = num_rules
        self.counts = np.zeros(num_rules)

    def map_stim_to_rule(self, dim: int, feature: int) -> int:
        if dim == 0:
            if feature == 0:
                return 0
            else:
                return 1
        else:
            if feature == 0:
                return 2
            else:
                return 3

    def map_rules_to_choices(self, stimulus):
        # given the stimulus, figure out which rules correspond to accepting (choice = 1) and which correspond to rejecting (choice = 0)
        # yes rules
        yes_rule1 = self.map_stim_to_rule(0, stimulus[0])
        yes_rule2 = self.map_stim_to_rule(1, stimulus[1])
        # no rules
        no_rule1 = self.map_stim_to_rule(0, 1 - stimulus[0])
        no_rule2 = self.map_stim_to_rule(1, 1 - stimulus[1])
        return (yes_rule1, yes_rule2), (no_rule1, no_rule2)

    def update(self, choice, obs, reward):
        # the choice has two dimensions and so we need to update both counts
        stim = obs[0]
        if choice == 0:
            stim = 1 - stim
        else:
            stim = stim
        rule1 = self.map_stim_to_rule(0, stim[0])
        rule2 = self.map_stim_to_rule(1, stim[1])
        if reward == 1:
            self.counts[rule1] += 1
            self.counts[rule2] += 1
        else:
            # get the other two rules
            other_rules = [x for x in range(4) if x != rule1 and x != rule2]
            self.counts[other_rules[0]] += 1
            self.counts[other_rules[1]] += 1

    def get_probabilities(self):
        return self.counts / np.sum(self.counts)

    def softmax(self):
        return np.exp(self.counts) / np.sum(np.exp(self.counts))

    def get_most_likely_rule(self):
        return np.argmax(self.counts)

    def act(self, obs):
        # use the counts array to make a choice based on the object in front of you
        # obs is tuple of (dim, feature), if most likely rule is 0 and obs has 0 in dim 0 return 1,
        # if most likely rule is 0 and obs has 1 in dim 0 return 0, etc.
        yes_rules, no_rules = self.map_rules_to_choices(obs[0])
        # use the probabilities of yes_rules and no_rules to make a choice
        probs = self.get_probabilities()
        yes_probs = probs[yes_rules[0]] + probs[yes_rules[1]]
        no_probs = probs[no_rules[0]] + probs[no_rules[1]]
        return np.random.choice([0, 1], p=[no_probs, yes_probs])

    def sim(self, env: WordsconsinEnv, num_trials: int) -> float:
        self.sim_rewards = np.zeros(num_trials)
        self.actions = np.zeros(num_trials)
        for i in range(num_trials):
            obs = env.get_obs()
            action = self.act(obs)
            self.actions[i] = action
            reward = env.step(action)[1]
            self.sim_rewards[i] = reward
        return np.sum(self.sim_rewards)


class expDecayModel(SimpleCountModel):
    """
    Updates the counts of each rule using an exponential decay.
    $$
    N_{k,t} = \exp(-\lambda) N_{k,t-1}
    $$

    Where $\lambda$ is the decay rate.
    """

    def __init__(self, num_rules, decay_rate=0.1):
        super().__init__(num_rules)
        self.decay_rate = decay_rate

    def update(self, choice, obs, reward):
        super().update(choice, obs, reward)
        self.counts = np.exp(-self.decay_rate) * self.counts
        # normalize properly
        self.counts = self.counts / np.sum(self.counts)


class stickyChoiceModel(SimpleCountModel):
    """
    $$
    p(c_t = k) =
    \begin{cases} 
    \eta + (1-\eta)\frac{N_k}{\sum_{j=1}^K N_j}, & k = c_{t-1} \\ 
    (1-\eta)\frac{N_{c_{t-1}}}{\sum_{j=1}^K N_j}, & k \neq c_{t-1}, k \leq K 
    \end{cases}
    $$
    """

    def __init__(self, num_rules, eta=0.5):
        super().__init__(num_rules)
        self.eta = eta
        self.prev_choice = None

    def act(self, obs):
        if self.prev_choice is None:
            pass
        # if we will choose the same rule as the last time then we use the first equation with \eta + (1-\eta) * N_k / sum(N)
        # otherwise we just use (1-\eta) * N_k / sum(N)


class CoinFlipModel(AbstractDecisionMaker):
    """
    Chooses 0 or 1 with probability $p$ and 1-p respectively. Just a coin flip to see how much better than random guessing we can do.
    """

    def __init__(self, p: float):
        self.p = p  # weight of coin flip

    def act(self, obs):
        return np.random.choice([0, 1], p=[1 - self.p, self.p])

    def sim(self, env: WordsconsinEnv, num_trials: int) -> float:
        self.sim_rewards = np.zeros(num_trials)
        self.actions = np.zeros(num_trials)
        for i in range(num_trials):
            obs = env.get_obs()
            action = self.act(obs)
            self.actions[i] = action
            reward = env.step(action)[1]
            self.sim_rewards[i] = reward
        return np.sum(self.sim_rewards)
