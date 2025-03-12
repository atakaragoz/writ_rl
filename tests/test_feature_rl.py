# test_feature_rl.py

import pytest
import numpy as np
from writ_tools.models.feature_rl import (
    FeatureRL,
    DecayFeatureRL,
    FeatureRLalphaMod,
    DecayFeatureRLalphaMod,
)
import pandas as pd
import numpy as np
import pandas as pd
import pytest
from writ_tools.models.feature_rl import FeatureRL


@pytest.fixture
def test_data():
    data = {
        "resp_numeric": [1, 1],
        "points": [1, 0],
        "item_rule_idx0": [0, 1],
        "item_rule_idx1": [0, 0],
        "inv_item_rule_idx0": [1, 0],
        "inv_item_rule_idx1": [1, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def feature_rl(test_data):
    model = FeatureRL(eta=0.1, beta=0.1, resp_st=0, data=test_data)
    model.value_chosen = 0.5
    return model


@pytest.fixture
def decay_feature_rl(test_data):

    model = DecayFeatureRL(
        eta=0.1, beta=0.1, kappa=0.1, decay=0.9, st=0, data=test_data
    )
    model.value_chosen = 0.5
    return model


@pytest.fixture
def feature_rlalpha_mod(test_data):
    model = FeatureRLalphaMod(eta=0.1, beta=0.1, kappa=0.1, resp_st=0, data=test_data)
    model.value_chosen = 0.5
    return model


@pytest.fixture
def decay_feature_rlalpha_mod(test_data):
    model = DecayFeatureRLalphaMod(
        eta=0.1, beta=0.1, kappa=0.1, decay=0.9, resp_st=0, data=test_data
    )
    model.value_chosen = 0.5
    return model


def test_feature_rlalpha_mod_update_weights(feature_rlalpha_mod):
    feature_rlalpha_mod.update_weights(stim_chosen_dim1=0, stim_chosen_dim2=1, reward=1)
    assert feature_rlalpha_mod.W[0, 0] != 0
    assert feature_rlalpha_mod.W[1, 1] != 0
    assert feature_rlalpha_mod.W[0, 1] == 0
    assert feature_rlalpha_mod.W[1, 0] == 0


def test_decay_feature_rlalpha_mod_update_weights(decay_feature_rlalpha_mod):
    decay_feature_rlalpha_mod.update_weights(
        stim_chosen_dim1=0, stim_chosen_dim2=1, reward=1
    )
    assert decay_feature_rlalpha_mod.W[0, 0] != 1
    assert decay_feature_rlalpha_mod.W[1, 1] != 1


# Test case for the construct_stimulus method
def test_construct_stimulus(feature_rl):
    expected_stimulus = np.array(
        [
            [[0, 1], [0, 0]],
            [[1, 0], [1, 1]],
        ]
    )
    assert np.array_equal(feature_rl.construct_stimulus(), expected_stimulus)


# Test case for the update_weights method
def test_update_weights(feature_rl):
    feature_rl.update_weights(stim_chosen_dim1=0, stim_chosen_dim2=1, reward=1)
    # self.value_chosen is originally 0.5 so the expected weights are 0.05
    expected_weights = np.array([[0.05, 0], [0, 0.05]])
    assert np.array_equal(feature_rl.W, expected_weights)


# Test case for the softmax method
def test_softmax(feature_rl):
    prob = feature_rl.softmax(0.5, 0.5)
    assert prob == 0.5


# Test case for the fit method
def test_fit(feature_rl):
    log_likelihood = feature_rl.fit()
    assert isinstance(log_likelihood, float)


if __name__ == "__main__":
    pytest.main()
