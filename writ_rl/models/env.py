import random
import numpy as np
from itertools import product
import gymnasium as gym
from gymnasium import spaces


class WordsconsinEnv(gym.Env):
    """
    Custom environment for the Wordsconsin game.

    Parameters:
    - num_blocks (int): Number of blocks in the game.
    - block_sizes (list): List of block sizes.
    - num_dimensions (int): Number of dimensions.
    - num_features (int): Number of features.
    - num_words (int): Number of words.

    Attributes:
    - num_blocks (int): Number of blocks in the game.
    - block_sizes (list): List of block sizes.
    - num_dimensions (int): Number of dimensions.
    - num_features (int): Number of features.
    - num_words (int): Number of words.
    - list_of_stims (list): List of stimulus vectors.
    - block_structure (list): List of block structures.
    - action_space (gym.Space): Action space for the environment.
    - observation_space (gym.Space): Observation space for the environment.
    - curr_block_idx (int): Index of the current block.
    - curr_word_idx (int): Index of the current word within the block.
    - curr_word (tuple): Current word.
    - learned_mat (dict): Dictionary representing the learned matrix.
    - discrim_map (dict): Dictionary representing the discrimination mapping.
    - non_discrim_map (dict): Dictionary representing the non-discrimination mapping.
    - targets (list): List of target words.

    Methods:
    - make_stim_list(): Generate the list of stimulus vectors.
    - step(action): Perform a step in the environment.
    - reset(): Reset the environment to its initial state.
    - make_block_structure(): Generate the block structure.
    - learned_and_discrim(): Initialize the learned and discrimination matrices.
    - fill_rest(): Fill the remaining words in the block structure.
    - assign_words(): Assign words to the block structure.
    - rule_mapping(rule, idx0, idx1): Map the rule to the correct choice.
    - make_correct_choice_vec(): Generate the correct choice vector.
    """

    def __init__(
        self, num_blocks, block_sizes, num_dimensions, num_features, num_words
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_sizes = block_sizes
        self.num_dimensions = num_dimensions
        self.num_features = num_features
        self.num_words = num_words
        self.list_of_stims = self.make_stim_list()
        self.block_structure = self.make_block_structure()
        if self.num_dimensions == 2 and self.num_features == 2:
            self.learned_and_discrim()
        self.provided_sub_block_structure = None
        # adding gym spaces for action and observation
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.MultiBinary(self.num_dimensions)

        self.curr_block_idx = 0  # keep track of current block
        self.curr_word_idx = 0  # keep track of current word within block
        self.curr_word = None  # store current word

    def make_stim_list(self):
        """
        Generate a list of stimuli vectors based on the number of dimensions and the number of words.

        Returns:
            list: A list of stimuli vectors, where each vector is a binary combination of values.
        """

        # Generate all possible combinations of binary values
        binary_combinations = list(product(range(2), repeat=self.num_dimensions))

        # Create a list of vectors for each binary combination
        list_of_stims = []
        for binary_combination in binary_combinations:
            curr_vectors = []
            for _ in range(self.num_words // len(binary_combinations)):
                curr_vectors.append(binary_combination)
            list_of_stims.append(curr_vectors)

        return list_of_stims

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and whether the episode is done.

        Parameters:
            action (int): The action taken by the agent.

        Returns:
            tuple: A tuple containing the next state (observation), reward, done flag, and additional information.

        """
        done = False
        reward = 0
        info = {}

        # Current word (observation)
        self.curr_word = self.block_structure[self.curr_block_idx]["words"][
            self.curr_word_idx
        ]

        # Determine if the action is correct
        correct_choice = self.rule_mapping(
            self.block_structure[self.curr_block_idx]["rule"],
            self.curr_word[0],
            self.curr_word[1],
        )

        # Assign reward based on action correctness
        if action == correct_choice:
            reward = 1

        # Prepare next observation for the next step
        next_word_idx = self.curr_word_idx + 1
        next_block_idx = self.curr_block_idx

        if next_word_idx >= len(self.block_structure[self.curr_block_idx]["words"]):
            next_word_idx = 0
            next_block_idx += 1

        if next_block_idx >= len(self.block_structure):
            done = True

        if not done:
            next_word = np.array(
                self.block_structure[next_block_idx]["words"][next_word_idx]
            )
        else:
            next_word = None
        # Update indices for the next step
        self.curr_word_idx = next_word_idx
        self.curr_block_idx = next_block_idx

        # Return the current observation, reward, and done flag
        return next_word, reward, done, info

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            numpy.ndarray: The initial state of the environment.
        """
        self.curr_block_idx = 0
        self.curr_word_idx = 0
        if self.provided_sub_block_structure is None:
            self.assign_words()
        return np.array(
            self.block_structure[self.curr_block_idx]["words"][self.curr_word_idx]
        )

    def make_block_structure(self):
        """
        Generates a block structure based on the given parameters.

        Returns:
            list: A list of dictionaries representing the block structure. Each dictionary contains the following keys:
                - "rule": The condition rule for the block.
                - "blockLen": The length of the block.
                - "words": An empty list to store words associated with the block.
        """

        conditions = list(range(1, self.num_dimensions * self.num_features + 1))
        num_conditions = len(conditions)
        num_blocks_per_condition = self.num_blocks // num_conditions

        block_list = []

        for condition in conditions:
            for _ in range(num_blocks_per_condition):
                for block_size in self.block_sizes:
                    block_list.append(
                        {"rule": condition, "blockLen": block_size, "words": []}
                    )

        # Shuffle blocks so that no two consecutive blocks have the same rule
        while True:
            random.shuffle(block_list)
            if all(
                block_list[i]["rule"] != block_list[i - 1]["rule"]
                for i in range(1, len(block_list))
            ):
                break

        return block_list

    def learned_and_discrim(self):
        """
        This method initializes the `learned_mat`, `discrim_map`, and `non_discrim_map` attributes.

        The `learned_mat` attribute is a nested dictionary that represents a matrix of learned values. Each key in the outer dictionary represents a state, and each key in the inner dictionary represents an action. The corresponding value is a list of integers representing the learned values for that state-action pair.

        The `discrim_map` attribute is a dictionary that maps each state to a list of states that are considered discriminative.

        The `non_discrim_map` attribute is a dictionary that maps each state to a list of states that are considered non-discriminative.
        """

        self.learned_mat = {
            1: {2: [0, 1, 2, 3], 3: [1, 2], 4: [0, 3]},
            2: {1: [0, 1, 2, 3], 3: [0, 3], 4: [1, 2]},
            3: {1: [1, 2], 2: [0, 3], 4: [0, 1, 2, 3]},
            4: {1: [0, 3], 2: [1, 2], 3: [0, 1, 2, 3]},
        }

        self.discrim_map = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
        self.non_discrim_map = {0: [0, 3], 1: [1, 2], 2: [1, 2], 3: [0, 3]}

    def fill_rest(self):
        """
        Fill the remaining slots in the block structure with words from the targets.

        This method iterates over the block structure and fills the remaining slots in each block
        with words randomly chosen from the targets. It ensures that each block has at least 2 words
        and that the chosen word comes from a non-empty target.

        Returns:
            None
        """
        for i in range(len(self.block_structure)):
            for j in range(2, self.block_structure[i]["blockLen"]):
                # Get the list of non-empty target indices
                non_empty_targets = [
                    index for index in range(4) if len(self.targets[index]) > 0
                ]

                # Randomly choose from the non-empty target indices
                rand_num = random.choice(non_empty_targets)

                self.block_structure[i]["words"].append(self.targets[rand_num].pop())

    def assign_words(self):
        """
        Assigns words to the block structure based on certain rules and conditions.

        This method shuffles the list of stimuli, divides it into two halves, and assigns the first half to the targets list.
        It then iterates over the block structure and assigns an appropriate word to each block based on the learned matrix,
        previous rule, current rule, and random indices. Additionally, it may assign a discriminating or non-discriminating word
        based on a random choice. Finally, it fills the remaining blocks with words from the targets list.

        Returns:
            None
        """

        self.targets = []

        for block in self.block_structure:
            block["words"] = []

        for stims in self.list_of_stims:
            random.shuffle(stims)
            half_length = len(stims) // 2
            self.targets.append(stims[:half_length])
        block_list = self.block_structure
        rand_num = random.randint(0, 3)
        # append the first two words for the first block at random
        self.block_structure[0]["words"].append(
            self.targets[random.randint(0, 3)].pop()
        )
        self.block_structure[0]["words"].append(
            self.targets[random.randint(0, 3)].pop()
        )
        for i in range(1, len(block_list)):
            previous_rule = rand_num + 1 if i == 0 else block_list[i - 1]["rule"]
            current_rule = block_list[i]["rule"]
            current_arr_len = len(self.learned_mat[previous_rule][current_rule])
            first_rand_idx = random.randint(0, current_arr_len - 1)
            discrim = random.choice([0, 1])
            second_rand_idx = random.choice([0, 1])

            self.block_structure[i]["words"].append(
                self.targets[
                    self.learned_mat[previous_rule][current_rule][first_rand_idx]
                ].pop()
            )

            if discrim:
                self.block_structure[i]["words"].append(
                    self.targets[
                        self.discrim_map[first_rand_idx][second_rand_idx]
                    ].pop()
                )
            else:
                self.block_structure[i]["words"].append(
                    self.targets[
                        self.non_discrim_map[first_rand_idx][second_rand_idx]
                    ].pop()
                )
        self.fill_rest()

    def rule_mapping(self, rule, idx0, idx1):
        """
        Maps the given rule to the corresponding index values.

        Parameters:
        - rule (int): The rule number.
        - idx0 (int): The value of the first index.
        - idx1 (int): The value of the second index.

        Returns:
        - int: The mapped value based on the rule and index values.
        """
        if rule == 1:
            return 1 if idx0 == 0 else 0
        elif rule == 2:
            return 1 if idx0 == 1 else 0
        elif rule == 3:
            return 1 if idx1 == 0 else 0
        elif rule == 4:
            return 1 if idx1 == 1 else 0

    def make_correct_choice_vec(self):
        # TODO this needs to be adjusted to instead compare stim to the correct rule for the block
        correct_choice_vec = []
        # for each word compare it to the rule for the block and append 1 if it matches and 0 if it doesn't
        for block in self.block_structure:
            for word in block["words"]:
                correct_choice_vec.append(
                    self.rule_mapping(block["rule"], word[0], word[1])
                )
        return correct_choice_vec
