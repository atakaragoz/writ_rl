from abc import ABC, abstractmethod
import numpy as np


class AbstractDecisionMaker(ABC):
    """
    Abstract base class for decision makers.

    All decision makers should inherit from this class. and contain the following methods:
    - act() -> int:
    - sim() -> float:
    """

    @abstractmethod
    def act(self, *args, **kwargs) -> int:
        """
        Make a decision based on the current state of the environment.

        Returns:
            int: The action to take.
        """
        pass

    @abstractmethod
    def sim(self, *args, **kwargs) -> float:
        """
        Simulate the decision maker over a number of trials.

        Returns:
            float: The total reward accumulated over the trials.
        """
        pass


class winStayLoseShift(AbstractDecisionMaker):
    def __init__(self, data=None):
        self.num_rules = 4
        self.active_rule = np.random.randint(self.num_rules)
        if data is not None:
            self.data = data
            self.n_trials = len(data)
            self.trial_by_trial_loglik = np.zeros((self.n_trials))
        self.loglik = 0

    def construct_stimulus(self) -> np.ndarray:
        """
        Constructs a stimulus array based on the data stored in the object.

        Returns:
            numpy.ndarray: The constructed stimulus array.

        """
        stim = np.array(
            (
                # self.data['word'].values,
                (
                    self.data["item_rule_idx0"].values,
                    self.data["item_rule_idx1"].values,
                ),
                (
                    self.data["inv_item_rule_idx0"].values,
                    self.data["inv_item_rule_idx1"].values,
                ),
            )
        )
        return stim

    def map_stim_to_rule(self, dim, feature):
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

    def update(self, choice, obs, reward):
        # the choice has two dimensions and so we need to update both counts
        if reward != 1:
            # switch to a random rule
            self.active_rule = np.random.choice(self.num_rules)

    def act(self, obs):
        if self.active_rule == 0:
            if obs[0][0] == 0:
                return 1
            else:
                return 0
        elif self.active_rule == 1:
            if obs[0][0] == 0:
                return 0
            else:
                return 1
        elif self.active_rule == 2:
            if obs[0][1] == 0:
                return 1
            else:
                return 0
        elif self.active_rule == 3:
            if obs[0][1] == 0:
                return 0
            else:
                return 1
        else:
            # choose at random [0 or 1]
            return np.random.choice([0, 1])

    def sim(self, env, n_trials):
        self.sim_rewards = np.zeros(n_trials)
        self.actions = np.zeros(n_trials)
        for i in range(n_trials):
            # choice = self.make_choice(env.get_obs())
            if i == 0:
                choice = np.random.choice([0, 1])
            else:
                choice = self.act(obs)
            obs = env.step(choice)
            self.update(choice, obs, obs[1])

            self.sim_rewards[i] = obs[1]
            self.actions[i] = choice
        return sum(self.sim_rewards)

    def finalize_loglik(self):
        if np.isnan(self.loglik):
            return np.inf
        return self.loglik


class winStayLoseShiftModelwithoutReplacement(winStayLoseShift):
    def __init__(self, data=None):
        super().__init__(data)
        self.num_rules = 4

    def update(self, choice, obs, reward):
        if reward != 1:
            # switch to a random rule without replacement (so don't use the current active rule)
            self.active_rule = np.random.choice(
                [x for x in range(self.num_rules) if x != self.active_rule]
            )


class winStayLoseShiftEpsilonGreedy(winStayLoseShift):
    def __init__(self, epsilon=0.1, data=None):
        super().__init__(data)
        self.num_rules = 4
        self.epsilon = epsilon

    def get_action_probabilities(self, obs):
        """
        returns the action probabilities for staying or switching.
        this is based on epsilon-greedy behavior.
        """
        base_choice = super().act(obs)  # deterministic wsls choice
        probabilities = np.full(
            self.num_rules, self.epsilon / self.num_rules
        )  # base exploration
        probabilities[base_choice] += (
            1 - self.epsilon
        )  # add probability to the preferred action
        self.P = probabilities
        return probabilities

    def act(self, obs):
        probabilities = self.get_action_probabilities(obs)
        return np.random.choice(self.num_rules, p=probabilities)

    def fit(self) -> float:
        stimulus_array = self.construct_stimulus()
        choices = self.data["resp_numeric"].values
        rewards = self.data["points"].values
        ticks = self.data["trial_within_block"].values

        for i in range(self.n_trials):
            if choices[i] == -1:
                R = np.array([0, 0])
                continue
            model_probs = self.get_action_probabilities(
                stimulus_array[:, :, i]
            )  # probabilities of each action
            choice_prob = model_probs[choices[i]]

            # update the loglik by comparing the result of act() with the choice that was made
            self.trial_by_trial_loglik[i] = np.log(choice_prob + 1e-8)
            self.loglik += self.trial_by_trial_loglik[i]
            self.update(choices[i], stimulus_array[:, :, i], rewards[i])

        return self.finalize_loglik()


class winStayLoseShiftEpsilonGreedyWithoutReplacement(winStayLoseShiftEpsilonGreedy):
    def __init__(self, epsilon=0.1, data=None):
        super().__init__(epsilon, data)
        self.num_rules = 4

    def update(self, choice, obs, reward):
        if reward != 1:
            # switch to a random rule without replacement (so don't use the current active rule)
            self.active_rule = np.random.choice(
                [x for x in range(self.num_rules) if x != self.active_rule]
            )
