import numpy as np
from scipy.optimize import minimize
import pandas as pd
from writ_tools.models.feature_rl import *
from writ_tools.models.RL_constants import MODELS
import multiprocessing as mp
import argparse
import os
import time
import hashlib


def worker_id_to_seed(worker_id, base_seed=691996):
    """
    Converts a string worker_id to an integer seed using hashing.

    Parameters:
    - worker_id (str): The string worker ID.
    - base_seed (int): The base seed for reproducibility.

    Returns:
    - int: An integer seed for random number generation.
    """
    # create a hash of the worker_id
    worker_hash = int(hashlib.sha256(worker_id.encode("utf-8")).hexdigest(), 16)

    # combine the base seed with the hashed worker_id to create the final seed
    return (worker_hash + base_seed) % (2**32)  # limit the seed to a 32-bit integer


# fit all subjects
def _MAP_fit(params, data, model, prior_distributions):
    """
    Fits a model using Maximum A Posteriori (MAP) estimation.

    Parameters:
    - params (list): List of model parameters.
    - data: The data used for fitting the model.
    - model: The model to be fitted.
    - prior_distributions (list): List of prior distributions for each parameter.

    Returns:
    - float: The log-likelihood of the fitted model.

    """

    log_likelihoods = []
    for i, param in enumerate(params):
        log_likelihoods.append(np.log(prior_distributions[i].pdf(param)))

    curr_model = model(*params, data)
    curr_model.fit()

    return curr_model.loglik + sum(log_likelihoods)


def fit_subject(args):
    """
    Fits the model to the data for a specific subject.

    Args:
        args (tuple): A tuple containing the following elements:
            - data (pandas.DataFrame): The data to fit the model to.
            - worker_id (int): The ID of the subject.
            - model (object): The model to fit.
            - prior_distributions (dict): The prior distributions for the model parameters.
            - bounds (list): The bounds for the model parameters.

    Returns:
        list: A list containing the following elements:
            - worker_id (int): The ID of the subject.
            - model parameters (list): The fitted model parameters.
            - negative log-likelihood (float): The negative log-likelihood of the fitted model.
    """
    data, worker_id, model, prior_distributions, bounds = args
    # convert worker_id to an integer seed
    seed = worker_id_to_seed(worker_id, 691996)

    # set the seed for this subject
    np.random.seed(seed)
    # get data for this subject
    data_sub = data[data["subid"] == worker_id]
    handle = lambda x: -_MAP_fit(x, data_sub, model, prior_distributions)
    best_res = None
    for _ in range(10):
        initial_params = [np.random.uniform(low, high) for low, high in bounds]

        res = minimize(handle, initial_params, bounds=bounds)

        if best_res is None or res.fun < best_res.fun:
            best_res = res
    # save
    result = [worker_id] + list(best_res.x) + [-best_res.fun]
    return result


def parallel_fit_all_subjects(data, model, prior_distributions, bounds):
    """
    Fits the model to the data for each subject in parallel using multiprocessing.

    Args:
        data (pd.DataFrame): The input data containing the subjects' information.
        model: The model to be fitted to the data.
        prior_distributions: The prior distributions for the model parameters.
        bounds: The bounds for the model parameters.

    Returns:
        pd.DataFrame: A dataframe containing the fitted parameters and log-likelihood for each subject.
    """
    # initialize
    num_params = len(bounds)

    # prepare arguments for each process
    args = [
        (data, worker_id, model, prior_distributions, bounds)
        for worker_id in data["subid"].unique()
    ]

    # create a pool of processes and map the function over the arguments
    with mp.Pool(10) as pool:
        results = pool.map(fit_subject, args)

    # make a nice dataframe
    columns = ["subid"] + [f"param_{i}" for i in range(num_params)] + ["loglik"]
    return pd.DataFrame(results, columns=columns)


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Filler description")
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path where the raw transcripts are stored",
    )
    parser.add_argument("-m", "--model", required=True, help="which model to fit"),
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Output path for where the models should be saved to",
    )
    parser.add_argument(
        "-r",
        "--no_resp_st",
        action="store_true",
        help="Whether to not include resp_st in the model",
        default=False,
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model.lower()
    resp_st = not args.no_resp_st
    assert model_name in MODELS.keys(), "Model not recognized!"
    assert os.path.exists(input_path), "Input path does not exist!"
    assert os.path.exists(output_path), "Output path does not exist!"
    df = pd.read_csv(input_path)
    model, prior_distributions, bounds, modelname, parameter_names = MODELS[model_name]
    if not resp_st:
        # the last bound is always stickiness so just force it to be (0,0)
        bounds[-1] = (0, 0)
        modelname += "_no_resp_st"
    results = parallel_fit_all_subjects(df, model, prior_distributions, bounds)

    # Add aic column to results
    if resp_st:
        results["aic"] = -2 * results["loglik"] + 2 * len(bounds)
    else:
        results["aic"] = -2 * results["loglik"] + 2 * (len(bounds) - 1)
    # Create a mapping dictionary
    rename_dict = {f"param_{i}": name for i, name in enumerate(parameter_names)}
    results["model"] = modelname
    # Rename the DataFrame columns
    results.rename(columns=rename_dict, inplace=True)
    if ".csv" in output_path:
        results.to_csv(output_path, index=False)
    else:
        results.to_csv(
            os.path.join(output_path, f"{modelname}_results.csv"), index=False
        )
    print(f"Finished fitting {modelname} in {time.time() - t0} seconds")


if __name__ == "__main__":
    np.random.seed(691996)  # setting random seed
    main()
