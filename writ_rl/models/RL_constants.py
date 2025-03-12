from writ_tools.models.feature_rl import *
from writ_tools.models.decisionmaker import (
    winStayLoseShiftEpsilonGreedy,
    winStayLoseShiftEpsilonGreedyWithoutReplacement,
)
from collections import namedtuple
import scipy.stats as ss

ModelInfo = namedtuple(
    "ModelInfo",
    ["model", "prior_distributions", "bounds", "modelname", "parameter_names"],
)

STANDARD_BETA_PRIOR_AND_BOUNDS = (ss.beta(2, 2), (1e-9, 1 - (1e-9)))

PARAMS = {
    "eta": STANDARD_BETA_PRIOR_AND_BOUNDS,
    "eta_neg": STANDARD_BETA_PRIOR_AND_BOUNDS,
    "eta_pos": STANDARD_BETA_PRIOR_AND_BOUNDS,
    "beta": (ss.gamma(4.82, scale=0.88), (1e-9, 10)),
    "decay": STANDARD_BETA_PRIOR_AND_BOUNDS,
    "kappa": STANDARD_BETA_PRIOR_AND_BOUNDS,
    "resp_st": (ss.norm(0, 1), (-4, 4)),
    "sop_eta": STANDARD_BETA_PRIOR_AND_BOUNDS,
    "lr_decay": STANDARD_BETA_PRIOR_AND_BOUNDS,
    "phi": STANDARD_BETA_PRIOR_AND_BOUNDS,
    "epsilon": (ss.beta(1, 1), (1e-9, 1 - (1e-9))),  # use an uninformative prior
    "pers": (ss.beta(1, 1), (1e-9, 1 - (1e-9))),
    "thresh": (ss.uniform(0, 3), (1e-9, 3 - (1e-9))),
}


def create_model_info(model_class, param_names, modelname):
    priors = [PARAMS[name][0] for name in param_names]
    bounds = [PARAMS[name][1] for name in param_names]
    return ModelInfo(model_class, priors, bounds, modelname, param_names)


MODELS = {
    "featurerl": create_model_info(FeatureRL, ["eta", "beta", "resp_st"], "FeatureRL"),
    "decayfeaturerl": create_model_info(
        DecayFeatureRL, ["eta", "beta", "decay", "resp_st"], "DecayFeatureRL"
    ),
    "decaythresh_featurerl": create_model_info(
        DecayThresholdFeatureRL,
        ["eta", "beta", "decay", "thresh", "resp_st"],
        "DecayThresholdFeatureRL",
    ),
    "decayduallearnfeaturerl": create_model_info(
        DecayDualLearningFeatureRL,
        ["eta_neg", "eta_pos", "beta", "decay", "resp_st"],
        "DecayDualLearnFeatureRL",
    ),
    "featurerlalphamod": create_model_info(
        FeatureRLalphaMod, ["eta", "beta", "kappa", "resp_st"], "FeatureRLalphaMod"
    ),
    "decayfeaturerlalphamod": create_model_info(
        DecayFeatureRLalphaMod,
        ["eta", "beta", "kappa", "decay", "resp_st"],
        "DecayFeatureRLalphaMod",
    ),
    "temporalcertainfeaturerl": create_model_info(
        TemporalCertainFeatureRL,
        ["eta", "beta", "resp_st"],
        "TemporalCertainFeatureRL",
    ),
    "decaytemporalcertainfeaturerl": create_model_info(
        DecayTemporalCertainFeatureRL,
        ["eta", "beta", "decay", "resp_st"],
        "DecayTemporalCertainFeatureRL",
    ),
    "decaytemporalcertainfeaturerl_tickp3": create_model_info(
        DecayTCFRL_tickp3,
        ["eta", "beta", "decay", "resp_st"],
        "DecayTemporalCertainFeatureRL_tickp3",
    ),
    "decaytemporalcertainfeaturerl_tickp5": create_model_info(
        DecayTCFRL_tickp5,
        ["eta", "beta", "decay", "resp_st"],
        "DecayTemporalCertainFeatureRL_tickp5",
    ),
    "decaytemporalcertainfeaturerl_tickuniform": create_model_info(
        DecayTCFRL_tickUniform,
        ["eta", "beta", "decay", "resp_st"],
        "DecayTemporalCertainFeatureRL_tickUniform",
    ),
    "decaytcfrl_ticku_noise": create_model_info(
        DecayTCFRL_tickUniformWithNoise,
        ["eta", "beta", "decay", "resp_st", "pers"],
        "DecayTemporalCertainFeatureRL_tickUniformWithNoise",
    ),
    "decaytemporalcertainfeaturerl_tickflip": create_model_info(
        DecayTCFRL_tickFlip,
        ["eta", "beta", "decay", "resp_st"],
        "DecayTemporalCertainFeatureRL_tickFlip",
    ),
    "selectiveattentionfeaturerl": create_model_info(
        SelectiveAttentionFeatureRL,
        ["eta", "beta", "phi", "resp_st"],
        "SelectiveAttentionFeatureRL",
    ),
    "decayselectiveattentionfeaturerl": create_model_info(
        DecaySelectiveAttentionFeatureRL,
        ["eta", "beta", "thresh", "decay", "resp_st"],
        "DecaySelectiveAttentionFeatureRL",
    ),
    "wslsepsilongreedy": create_model_info(
        winStayLoseShiftEpsilonGreedy,
        ["epsilon"],
        "WinStayLoseShiftEpsilonGreedy",
    ),
    "wslsepsilongreedywithoutreplacement": create_model_info(
        winStayLoseShiftEpsilonGreedyWithoutReplacement,
        ["epsilon"],
        "WinStayLoseShiftEpsilonGreedyWithoutReplacement",
    ),
}
