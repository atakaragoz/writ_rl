import numpy as np
from .env import WordsconsinEnv
from typing import Tuple
from .decisionmaker import AbstractDecisionMaker


class FeatureRL(AbstractDecisionMaker):
    """
    Feature-based RL class.
    Parameters:
    - eta (float): Learning rate.
    - beta (float): Inverse temperature parameter.
    - st (float): Response stickiness.
    - data (numpy.ndarray): Input data.
    Attributes:
    - eta (float): Learning rate.
    - beta (float): Inverse temperature parameter.
    - st (float): Response stickiness.
    - data (numpy.ndarray): Input data.
    - W (numpy.ndarray): Weight matrix.
    - Ws (numpy.ndarray): Weight matrix history.
    - Vs (numpy.ndarray): Value history.
    - n_trials (int): Number of trials.
    - rpe (numpy.ndarray): Reward prediction error history.
    - loglik (float): Log-likelihood.
    - trial_by_trial_loglik (numpy.ndarray): Trial-by-trial log-likelihood history.
    - value_chosen (float): Value of the chosen stimulus.
    - value_not_chosen (float): Value of the not chosen stimulus.
    - uncertainties (numpy.ndarray): Uncertainty history.
    - P (numpy.ndarray): Probability distribution.
    Methods:
    - construct_stimulus(): Construct the stimulus array.
    - update_weights(stim_chosen_dim1, stim_chosen_dim2, reward): Update the weight matrix.
    - softmax(value_chosen, value_not_chosen): Compute the softmax probability.
    - sim(): Simulate the RL feature.
    - uncertainty(): Compute the uncertainty using KL divergence.
    - fit(): Fit the RL feature model.
    """

    def __init__(self, eta: float, beta: float, resp_st: float, data=None):
        self.eta = eta
        self.beta = beta
        self.st = resp_st  # response stickiness
        if data is not None:
            self.data = data
            self.Ws = np.zeros((2, 2, self.data.shape[0]))
            self.Vs = np.zeros((self.data.shape[0]))
            self.n_trials = len(data)
            self.rpe = np.zeros((self.n_trials))
            self.trial_by_trial_loglik = np.zeros((self.n_trials))
            self.uncertainties = np.zeros((self.n_trials))

        self.W = np.zeros((2, 2))
        self.loglik = 0
        self.value_chosen = 0.5
        self.value_not_chosen = 0.5
        self.P = np.array([0.25, 0.25, 0.25, 0.25])

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

    def update_weights(
        self, stim_chosen_dim1: int, stim_chosen_dim2: int, reward: float
    ) -> None:
        """
        Update the weights of the RL agent based on the chosen stimuli and reward.

        Parameters:
        - stim_chosen_dim1: The index of the chosen stimulus in dimension 1.
        - stim_chosen_dim2: The index of the chosen stimulus in dimension 2.
        - reward: The reward received for choosing the stimuli.

        Returns:
        None
        """
        self.W[0, stim_chosen_dim1] += self.eta * (reward - self.value_chosen)
        self.W[1, stim_chosen_dim2] += self.eta * (reward - self.value_chosen)
        # Convert W matrix to probabilities
        flattened_w = self.W.ravel()
        exp_w = np.exp(flattened_w - np.max(flattened_w))  # softmax
        self.P = exp_w / exp_w.sum()

    def softmax(self, value_chosen: float, value_not_chosen: float) -> float:
        """
        Calculates the softmax function for two given values.

        Parameters:
        - value_chosen: The value associated with the chosen option.
        - value_not_chosen: The value associated with the not chosen option.

        Returns:
        - The softmax probability of choosing the value_chosen option.

        """
        max_value = np.max([value_chosen, value_not_chosen])
        exp_values = np.exp(
            self.beta
            * np.array([value_chosen - max_value, value_not_chosen - max_value])
        )
        return exp_values[0] / np.sum(exp_values)

    def act(
        self, value_choose_yes: float, value_choose_no: float, R: np.ndarray
    ) -> int:
        action_prob = self.softmax(
            value_choose_yes + self.st * R[1], value_choose_no + self.st * R[0]
        )
        action = np.random.choice([0, 1], p=[1 - action_prob, action_prob])
        if action == 0:
            self.value_chosen = value_choose_no
            self.value_not_chosen = value_choose_yes
        else:
            self.value_chosen = value_choose_yes
            self.value_not_chosen = value_choose_no
        return action

    def sim(self, env: WordsconsinEnv, n_trials: int) -> float:
        self.sim_rewards = np.zeros((n_trials))
        self.actions = np.zeros((n_trials))
        self.Ws = np.zeros((2, 2, n_trials))
        self.Vs = np.zeros((n_trials))
        self.uncertainties = np.zeros((n_trials))
        i = 0
        done = False
        while not done:
            if i == 0:
                obs = (
                    env.reset()
                )  # first observation is an array of the form dim 0 of stim and dim 1 of stim

            R = self.get_response_stickiness(i, self.actions)
            choose_yes_dim1, choose_yes_dim2 = obs
            choose_no_dim1, choose_no_dim2 = 1 - choose_yes_dim1, 1 - choose_yes_dim2
            value_choose_yes, value_choose_no = self.calculate_values(
                choose_yes_dim1, choose_yes_dim2, choose_no_dim1, choose_no_dim2
            )
            action = self.act(value_choose_yes, value_choose_no, R)
            self.actions[i] = action
            if action == 0:
                stim_chosen_dim1 = choose_no_dim1
                stim_chosen_dim2 = choose_no_dim2
            else:
                stim_chosen_dim1 = choose_yes_dim1
                stim_chosen_dim2 = choose_yes_dim2
            obs, reward, done, info = env.step(action)

            self.update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)
            self.update_histories(i)
            self.sim_rewards[i] = reward
            i += 1
        return sum(self.sim_rewards)

    def handle_missing_choice(self, i: int) -> None:
        self.rpe[i] = np.nan

    def get_response_stickiness(self, i: int, choices: np.ndarray) -> np.ndarray:
        if i == 0:
            return np.array([0, 0])
        elif choices[i] == choices[i - 1]:
            if choices[i] == 1:
                return np.array([1, 0])
            else:  # choices[i] == 0
                return np.array([1, 0])
        elif choices[i] != choices[i - 1]:
            if choices[i] == 1:
                return np.array([0, 1])
            else:
                return np.array([0, 1])

    def fit(self):
        stimulus_array = self.construct_stimulus()
        choices = self.data["resp_numeric"].values
        rewards = self.data["points"].values

        for i in range(self.n_trials):
            if choices[i] == -1:
                self.handle_missing_choice(i)
                R = np.array([0, 0])
                continue
            R = self.get_response_stickiness(i, choices)
            (
                stim_chosen_dim1,
                stim_chosen_dim2,
                stim_not_chosen_dim1,
                stim_not_chosen_dim2,
            ) = self.get_stimuli_dimensions(stimulus_array, choices, i)
            self.process_trial(
                i,
                stim_chosen_dim1,
                stim_chosen_dim2,
                stim_not_chosen_dim1,
                stim_not_chosen_dim2,
                rewards[i],
                R,
            )

        return self.finalize_loglik()

    def process_trial(
        self,
        i: int,
        stim_chosen_dim1: int,
        stim_chosen_dim2: int,
        stim_not_chosen_dim1: int,
        stim_not_chosen_dim2: int,
        reward: float,
        R: np.ndarray,
    ):
        self.calculate_values(
            stim_chosen_dim1,
            stim_chosen_dim2,
            stim_not_chosen_dim1,
            stim_not_chosen_dim2,
        )
        self.update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)
        self.update_histories(i)
        self.update_loglik_and_uncertainties(i, R, reward)

    def calculate_values(
        self,
        stim_chosen_dim1: int,
        stim_chosen_dim2: int,
        stim_not_chosen_dim1: int,
        stim_not_chosen_dim2: int,
    ) -> Tuple[float, float]:
        value_chosen = self.W[0, stim_chosen_dim1] + self.W[1, stim_chosen_dim2]
        value_not_chosen = (
            self.W[0, stim_not_chosen_dim1] + self.W[1, stim_not_chosen_dim2]
        )
        self.value_chosen = value_chosen
        self.value_not_chosen = value_not_chosen
        return value_chosen, value_not_chosen

    def update_histories(self, i: int) -> None:
        self.Ws[:, :, i] = self.W
        self.Vs[i] = self.value_chosen
        self.uncertainties[i] = self.uncertainty()

    def uncertainty(self) -> float:
        # calculates KL divergence of current W matrix from a uniform distribution
        uniform_dist = np.array([0.25, 0.25, 0.25, 0.25])
        kl_div = np.sum(self.P * np.log(self.P / uniform_dist))

        return kl_div

    def update_loglik_and_uncertainties(
        self, i: int, R: np.ndarray, reward: float
    ) -> None:
        p_chosen = self.softmax(
            self.value_chosen + self.st * R[0], self.value_not_chosen + self.st * R[1]
        )
        self.loglik += np.log(p_chosen)
        self.rpe[i] = reward - self.value_chosen
        self.trial_by_trial_loglik[i] = np.log(p_chosen)
        self.uncertainties[i] = self.uncertainty()
        # self.uncertainties[i] = 0

    def get_stimuli_dimensions(
        self, stimulus_array: np.ndarray, choices: np.ndarray, i: int
    ) -> Tuple[int, int, int, int]:
        stim_chosen_dim1 = stimulus_array[choices[i]][0][i]
        stim_chosen_dim2 = stimulus_array[choices[i]][1][i]
        stim_not_chosen_dim1 = 1 - stim_chosen_dim1
        stim_not_chosen_dim2 = 1 - stim_chosen_dim2
        return (
            stim_chosen_dim1,
            stim_chosen_dim2,
            stim_not_chosen_dim1,
            stim_not_chosen_dim2,
        )

    def finalize_loglik(self):
        if np.isnan(self.loglik):
            return np.inf
        return self.loglik


class DecayFeatureRL(FeatureRL):
    """
    A reinforcement learning class that implements feature-based learning with decay for the unchosen features.

    Parameters:
    - eta (float): The learning rate.
    - beta (float): The exploration rate.
    - decay (float): The rate at which the value of non-chosen features decays.
    - st (object): The state object.
    - data (object): The data object.

    Attributes:
    - decay (float): The rate at which the value of non-chosen features decays.

    Methods:
    - update_weights(stim_chosen_dim1, stim_chosen_dim2, reward): Updates the weights based on the chosen stimuli and reward.

    Inherits from:
    - FeatureRL: A base class for feature-based reinforcement learning.
    """

    def __init__(
        self, eta: float, beta: float, decay: float, resp_st: float, data=None
    ):
        super().__init__(eta, beta, resp_st, data)
        self.decay = decay  # rate at which the value of non-chosen feature decays

    def update_weights(
        self, stim_chosen_dim1: int, stim_chosen_dim2: int, reward: float
    ) -> None:
        super().update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)
        stim_not_chosen_dim1 = 1 - stim_chosen_dim1
        stim_not_chosen_dim2 = 1 - stim_chosen_dim2

        self.W[0, stim_not_chosen_dim1] *= self.decay
        self.W[1, stim_not_chosen_dim2] *= self.decay


class DecayDualLearningFeatureRL(DecayFeatureRL):
    def __init__(
        self,
        eta_neg: float,
        eta_pos: float,
        beta: float,
        decay: float,
        resp_st: float,
        data=None,
    ):
        super().__init__(eta_neg, beta, decay, resp_st, data)
        self.eta_neg = eta_neg
        self.eta_pos = eta_pos

    def update_weights(self, stim_chosen_dim1, stim_chosen_dim2, reward):
        if reward - self.value_chosen <= 0:
            self.eta = self.eta_neg
        elif reward - self.value_chosen > 0:
            self.eta = self.eta_pos

        self.W[0, stim_chosen_dim1] += self.eta * (reward - self.value_chosen)
        self.W[1, stim_chosen_dim2] += self.eta * (reward - self.value_chosen)
        # Convert W matrix to probabilities
        flattened_w = self.W.ravel()
        exp_w = np.exp(flattened_w - np.max(flattened_w))  # softmax
        self.P = exp_w / exp_w.sum()


class DecayThresholdFeatureRL(DecayFeatureRL):
    def __init__(
        self,
        eta: float,
        beta: float,
        decay: float,
        threshold: float,
        resp_st: float,
        data=None,
    ):
        super().__init__(eta, beta, decay, resp_st, data)
        self.thresh = threshold

    def update_weights(self, stim_chosen_dim1, stim_chosen_dim2, reward):
        super().update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)
        if abs(reward - self.value_chosen) > self.thresh:
            self.weights = np.ones((2, 2)) * 0.25


class FeatureRLalphaMod(FeatureRL):
    """
    A reinforcement learning model that incorporates a modulated learning rate based on the unsigned reward prediction error (RPE).

    Parameters:
    - eta (float): The base learning rate.
    - beta (float): The discount factor for future rewards.
    - kappa (float): The modulation factor for the learning rate.
    - st (float): The softmax temperature parameter.
    - data (list): The input data for training the model.

    Attributes:
    - kappa (float): The modulation factor for the learning rate.

    Methods:
    - update_weights(stim_chosen_dim1, stim_chosen_dim2, reward): Updates the weights of the model based on the chosen stimuli and reward.
    this model is based on Rouhani and Niv 2021 where the unsigned RPE modulates the learning rate
    """

    def __init__(
        self, eta: float, beta: float, kappa: float, resp_st: float, data=None
    ):
        super().__init__(eta, beta, resp_st, data)
        self.kappa = kappa

    def update_weights(
        self, stim_chosen_dim1: int, stim_chosen_dim2: int, reward: float
    ) -> None:
        """
        Update the weights of the RL agent based on the chosen stimuli and reward.

        Args:
            stim_chosen_dim1 (int): The index of the chosen stimulus in dimension 1.
            stim_chosen_dim2 (int): The index of the chosen stimulus in dimension 2.
            reward (float): The reward received for the chosen stimuli.

        Returns:
            None
        """
        # need to first get the unsigned rpe
        rpe = reward - self.value_chosen
        unsigned_rpe = np.abs(rpe)
        # now we update the alpha
        self.alpha = self.eta * self.kappa * unsigned_rpe
        # we need to make sure self.alpha is between 0 and 1 so we use a sigmoid
        self.alpha = 1 / (1 + np.exp(-self.alpha))
        # now we update the weightsq
        self.W[0, stim_chosen_dim1] += self.alpha * (reward - self.value_chosen)
        self.W[1, stim_chosen_dim2] += self.alpha * (reward - self.value_chosen)


class DecayFeatureRLalphaMod(FeatureRLalphaMod):
    """
    A class representing a modified version of FeatureRLalphaMod with decayed non-chosen feature values.

    Parameters:
    - eta (float): The learning rate.
    - beta (float): The exploration rate.
    - kappa (float): The eligibility trace decay rate.
    - decay (float): The rate at which the value of non-chosen feature decays.
    - st (object): The state object.
    - data (object): The data object.

    Attributes:
    - decay (float): The rate at which the value of non-chosen feature decays.

    Methods:
    - update_weights(stim_chosen_dim1, stim_chosen_dim2, reward): Updates the weights based on the chosen stimuli and reward.

    Inherits from:
    - FeatureRLalphaMod
    """

    def __init__(
        self,
        eta: float,
        beta: float,
        kappa: float,
        decay: float,
        resp_st: float,
        data=None,
    ):
        super().__init__(eta, beta, kappa, resp_st, data)
        self.decay = decay  # rate at which the value of non-chosen feature decays

    def update_weights(
        self, stim_chosen_dim1: int, stim_chosen_dim2: int, reward: float
    ) -> None:
        super().update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)
        stim_not_chosen_dim1 = 1 - stim_chosen_dim1
        stim_not_chosen_dim2 = 1 - stim_chosen_dim2

        self.W[0, stim_not_chosen_dim1] *= self.decay
        self.W[1, stim_not_chosen_dim2] *= self.decay


class TemporalCertainFeatureRL(FeatureRL):
    def __init__(self, eta: float, beta: float, resp_st: float, data=None):
        super().__init__(eta, beta, resp_st, data)

    def check_tick(self, tick: int) -> None:
        if tick == 0:
            old_rule = np.argmax(self.W)
            self.W.ravel()[old_rule] = 0
            secondary = np.argmax(self.W)

            # set W to 0 at the old rule and 0.333 at every other location
            self.W = np.ones((2, 2)) * 0.5
            self.W.ravel()[old_rule] = 0
            self.W.ravel()[secondary] = 0
            # self.W *= -1

    def sim(self, env: WordsconsinEnv, n_trials: int) -> float:
        self.sim_rewards = np.zeros((n_trials))
        self.actions = np.zeros((n_trials))
        self.Ws = np.zeros((2, 2, n_trials))
        self.Vs = np.zeros((n_trials))
        self.uncertainties = np.zeros((n_trials))
        i = 0
        done = False
        trial_within_block = [np.arange(0, x["blockLen"]) for x in env.block_structure]
        trial_within_block = [
            item for sublist in trial_within_block for item in sublist
        ]
        while not done:
            if i == 0:
                obs = env.reset()
            if i != 0:
                self.check_tick(trial_within_block[i])
            R = self.get_response_stickiness(i, self.actions)
            if trial_within_block[i] == 0:
                R = np.array([0, 0])
            choose_yes_dim1, choose_yes_dim2 = obs
            choose_no_dim1, choose_no_dim2 = 1 - choose_yes_dim1, 1 - choose_yes_dim2
            value_choose_yes, value_choose_no = self.calculate_values(
                choose_yes_dim1, choose_yes_dim2, choose_no_dim1, choose_no_dim2
            )
            action = self.act(value_choose_yes, value_choose_no, R)
            self.actions[i] = action
            if action == 0:
                stim_chosen_dim1 = choose_no_dim1
                stim_chosen_dim2 = choose_no_dim2
            else:
                stim_chosen_dim1 = choose_yes_dim1
                stim_chosen_dim2 = choose_yes_dim2

            obs, reward, done, info = env.step(action)

            self.update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)

            self.update_histories(i)
            self.sim_rewards[i] = reward
            i += 1
        return sum(self.sim_rewards)

    def fit(self):
        stimulus_array = self.construct_stimulus()
        choices = self.data["resp_numeric"].values
        rewards = self.data["points"].values
        ticks = self.data["trial_within_block"].values

        for i in range(self.n_trials):
            if choices[i] == -1:
                R = np.array([0, 0])
                self.handle_missing_choice(i)
                continue

            R = self.get_response_stickiness(i, choices)
            (
                stim_chosen_dim1,
                stim_chosen_dim2,
                stim_not_chosen_dim1,
                stim_not_chosen_dim2,
            ) = self.get_stimuli_dimensions(stimulus_array, choices, i)
            self.process_trial(
                i,
                stim_chosen_dim1,
                stim_chosen_dim2,
                stim_not_chosen_dim1,
                stim_not_chosen_dim2,
                rewards[i],
                R,
                ticks[i],
            )

        return self.finalize_loglik()

    def process_trial(
        self,
        i: int,
        stim_chosen_dim1: int,
        stim_chosen_dim2: int,
        stim_not_chosen_dim1: int,
        stim_not_chosen_dim2: int,
        reward: float,
        R: np.ndarray,
        tick: int,
    ) -> None:
        self.check_tick(tick)
        self.calculate_values(
            stim_chosen_dim1,
            stim_chosen_dim2,
            stim_not_chosen_dim1,
            stim_not_chosen_dim2,
        )
        self.update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)
        self.update_histories(i)
        self.update_loglik_and_uncertainties(i, R, reward)


class DecayTemporalCertainFeatureRL(TemporalCertainFeatureRL):
    def __init__(
        self, eta: float, beta: float, decay: float, resp_st: float, data=None
    ):
        super().__init__(eta, beta, resp_st, data)
        self.decay = decay

    def update_weights(
        self, stim_chosen_dim1: int, stim_chosen_dim2: int, reward: float
    ) -> None:
        super().update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)
        stim_not_chosen_dim1 = 1 - stim_chosen_dim1
        stim_not_chosen_dim2 = 1 - stim_chosen_dim2

        self.W[0, stim_not_chosen_dim1] *= self.decay
        self.W[1, stim_not_chosen_dim2] *= self.decay


class DecayTCFRL_tickp3(DecayTemporalCertainFeatureRL):
    def __init__(
        self, eta: float, beta: float, decay: float, resp_st: float, data=None
    ):
        super().__init__(eta, beta, decay, resp_st, data)

    def check_tick(self, tick: int) -> None:
        if tick == 0:
            old_rule = np.argmax(self.W)
            # set W to 0 at the old rule and 0.333 at every other location
            self.W = np.ones((2, 2)) * 0.333
            self.W.ravel()[old_rule] = 0
            # self.W *= -1


class DecayTCFRL_tickp5(DecayTemporalCertainFeatureRL):
    def __init__(
        self, eta: float, beta: float, decay: float, resp_st: float, data=None
    ):
        super().__init__(eta, beta, decay, resp_st, data)

    def check_tick(self, tick: int) -> None:
        if tick == 0:
            old_rule = np.argmax(self.W)
            self.W.ravel()[old_rule] = 0
            secondary = np.argmax(self.W)
            self.W.ravel()[secondary] = 0
            self.W = np.ones((2, 2)) * 0.5
            self.W.ravel()[old_rule] = 0
            self.W.ravel()[secondary] = 0


class DecayTCFRL_tickUniform(DecayTemporalCertainFeatureRL):
    def __init__(
        self, eta: float, beta: float, decay: float, resp_st: float, data=None
    ):
        super().__init__(eta, beta, decay, resp_st, data)

    def check_tick(self, tick: int) -> None:
        if tick == 0:
            self.W = np.ones((2, 2)) * 0.25


class DecayTCFRL_tickUniformWithNoise(DecayTemporalCertainFeatureRL):
    def __init__(
        self,
        eta: float,
        beta: float,
        decay: float,
        resp_st: float,
        pers: float,
        data=None,
    ):
        super().__init__(eta, beta, decay, resp_st, data)
        self.pers = pers  # a value between 0 and 1 that determines whether the tickflip properly happens to account for some perseverance

    def check_tick(self, tick):
        if tick == 0 and np.random.rand() < self.pers:
            self.W = np.ones((2, 2)) * 0.25


class DecayTCFRL_tickFlip(DecayTemporalCertainFeatureRL):
    def __init__(
        self, eta: float, beta: float, decay: float, resp_st: float, data=None
    ):
        super().__init__(eta, beta, decay, resp_st, data)

    def check_tick(self, tick: int) -> None:
        if tick == 0:
            # self.W = np.ones((2, 2)) * 0.25
            self.W *= -1


class SelectiveAttentionFeatureRL(FeatureRL):
    def __init__(self, eta: float, beta: float, phi: float, resp_st: float, data=None):
        super().__init__(eta, beta, resp_st, data)
        self.phi = phi
        self.A = np.ones((2, 2)) / 4

    def update_attention(self, corr_rule_numeric: int) -> None:
        self.A = np.zeros((2, 2))
        self.A.ravel()[corr_rule_numeric] = 1
        self.A[self.A == 0] = 1e-3
        self.W = self.A

    def sim(self, env: WordsconsinEnv, n_trials: int) -> float:
        self.sim_rewards = np.zeros((n_trials))
        self.actions = np.zeros((n_trials))
        self.Ws = np.zeros((2, 2, n_trials))
        self.Vs = np.zeros((n_trials))
        self.uncertainties = np.zeros((n_trials))
        corr_rule_numeric = [[x["rule"]] * x["blockLen"] for x in env.block_structure]
        corr_rule_numeric = [item for sublist in corr_rule_numeric for item in sublist]
        trial_within_block = [np.arange(0, x["blockLen"]) for x in env.block_structure]
        trial_within_block = [
            item for sublist in trial_within_block for item in sublist
        ]
        i = 0
        done = False
        while not done:
            if i == 0:
                obs = env.reset()

            R = self.get_response_stickiness(i, self.actions)
            #     self.update_attention(corr_rule_numeric[i] - 1)
            choose_yes_dim1, choose_yes_dim2 = obs
            choose_no_dim1, choose_no_dim2 = 1 - choose_yes_dim1, 1 - choose_yes_dim2
            value_choose_yes, value_choose_no = self.calculate_values(
                choose_yes_dim1, choose_yes_dim2, choose_no_dim1, choose_no_dim2
            )
            action = self.act(value_choose_yes, value_choose_no, R)
            if action == 0:
                stim_chosen_dim1 = choose_no_dim1
                stim_chosen_dim2 = choose_no_dim2
                self.value_chosen = value_choose_no
                self.value_not_chosen = value_choose_yes
            else:
                stim_chosen_dim1 = choose_yes_dim1
                stim_chosen_dim2 = choose_yes_dim2
                self.value_chosen = value_choose_yes
                self.value_not_chosen = value_choose_no

            obs, reward, done, info = env.step(action)
            rpe = reward - self.value_chosen
            if np.abs(rpe) >= self.phi:
                self.update_attention(corr_rule_numeric[i])

            self.update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)

            self.update_histories(i)
            self.sim_rewards[i] = reward
            i += 1
        return sum(self.sim_rewards)

    def fit(self):
        stimulus_array = self.construct_stimulus()
        choices = self.data["resp_numeric"].values
        rewards = self.data["points"].values
        corr_rule_numerics = self.data["corr_rule_numeric"].values
        trial_within_block = self.data["trial_within_block"].values

        for i in range(self.n_trials):
            if choices[i] == -1:
                self.handle_missing_choice(i)
                R = np.array([0, 0])
                continue
            # if trial_within_block[i] == 0:
            #     self.update_attention(corr_rule_numerics[i])

            R = self.get_response_stickiness(i, choices)
            (
                stim_chosen_dim1,
                stim_chosen_dim2,
                stim_not_chosen_dim1,
                stim_not_chosen_dim2,
            ) = self.get_stimuli_dimensions(stimulus_array, choices, i)
            # assuming perfect knowledge of the correct rule

            self.process_trial(
                i,
                stim_chosen_dim1,
                stim_chosen_dim2,
                stim_not_chosen_dim1,
                stim_not_chosen_dim2,
                rewards[i],
                R,
            )
            if np.abs(self.rpe[i]) >= self.phi:
                self.update_attention(corr_rule_numerics[i])

        return self.finalize_loglik()

    def process_trial(
        self,
        i: int,
        stim_chosen_dim1: int,
        stim_chosen_dim2: int,
        stim_not_chosen_dim1: int,
        stim_not_chosen_dim2: int,
        reward: float,
        R: np.ndarray,
    ):
        self.calculate_values(
            stim_chosen_dim1,
            stim_chosen_dim2,
            stim_not_chosen_dim1,
            stim_not_chosen_dim2,
        )
        self.update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)
        self.update_histories(i)
        self.update_loglik_and_uncertainties(i, R, reward)


class DecaySelectiveAttentionFeatureRL(SelectiveAttentionFeatureRL):
    def __init__(
        self,
        eta: float,
        beta: float,
        phi: float,
        decay: float,
        resp_st: float,
        data=None,
    ):
        super().__init__(eta, beta, phi, resp_st, data)
        self.decay = decay

    def update_weights(
        self, stim_chosen_dim1: int, stim_chosen_dim2: int, reward: float
    ) -> None:
        super().update_weights(stim_chosen_dim1, stim_chosen_dim2, reward)
        stim_not_chosen_dim1 = 1 - stim_chosen_dim1
        stim_not_chosen_dim2 = 1 - stim_chosen_dim2

        self.W[0, stim_not_chosen_dim1] *= self.decay
        self.W[1, stim_not_chosen_dim2] *= self.decay
