import numpy as np
import pandas as pd
from writ_tools.data.process_strat import *
from writ_tools.models.feature_rl import *
from writ_tools.models.RL_constants import MODELS
import scipy.stats as ss


def generate_env(env_obj, env_params):
    env1 = env_obj(*env_params)
    env1.reset()
    corr_rule_numeric = [[x["rule"]] * x["blockLen"] for x in env1.block_structure]
    corr_rule_numeric = [item for sublist in corr_rule_numeric for item in sublist]

    block_id = [[i] * x["blockLen"] for i, x in enumerate(env1.block_structure)]
    block_id = [item for sublist in block_id for item in sublist]

    trial_within_block = [np.arange(0, x["blockLen"]) for x in env1.block_structure]
    trial_within_block = [item for sublist in trial_within_block for item in sublist]

    trial_num = np.arange(0, len(corr_rule_numeric))
    return env1, corr_rule_numeric, block_id, trial_within_block, trial_num


def select_initial_params(model):
    bounds = MODELS[model].bounds

    return [np.random.uniform(low, high) for low, high in bounds], None


def init_model_for_sim(model_obj, params=None):
    if params is None:
        init_params = select_initial_params(model_obj)[0]
    else:
        init_params = params
    param_names = MODELS[model_obj].parameter_names
    params = dict(zip(param_names, init_params))
    params["resp_st"] = ss.norm(0, 1).rvs()
    model = MODELS[model_obj].model(**params, data=None)
    return model, params


def sim_model(number_sims, model_name, env_obj, env_params, n_trials):
    env1, corr_rule_numeric, block_id, trial_within_block, trial_num = generate_env(
        env_obj, env_params
    )
    sim_data = pd.DataFrame()
    for i in range(number_sims):
        model, params = init_model_for_sim(model_name)
        model.sim(env1, n_trials)
        sim_data_sub = pd.DataFrame()

        sim_data_sub["block_id"] = block_id
        sim_data_sub["trial_within_block"] = trial_within_block
        sim_data_sub["trial_num"] = trial_num
        sim_data_sub["corr_rule_numeric"] = corr_rule_numeric
        sim_data_sub["action"] = model.actions
        sim_data_sub["reward"] = model.sim_rewards
        sim_data_sub["subid"] = str(i + 1)
        for param_name, param in params.items():
            sim_data_sub[param_name] = param
        sim_data = pd.concat([sim_data, sim_data_sub])
    return sim_data.reset_index(drop=True)


def process_sim_data(sim_data):
    sim_data["points"] = sim_data["reward"]
    sim_data = (
        sim_data.groupby("subid")
        .apply(lambda x: label_item_relative_to_index(label_strat_blocks(x)))
        .reset_index(drop=True)
    )
    return sim_data
