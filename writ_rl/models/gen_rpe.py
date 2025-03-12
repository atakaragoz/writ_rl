# this code will combine the fitted model values for a given participant with the trials they experienced during encoding to get an RPE estimate for each trial

import numpy as np
import pandas as pd
from writ_tools.models.feature_rl import *
import argparse
import os
import time
from writ_tools.models.RL_constants import MODELS


def load_data(input_path):
    """
    Load data from a CSV file.

    Parameters:
    input_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(input_path)


def get_single_subject_data(data, worker_id):
    """
    Retrieves the data for a single subject based on their worker ID.

    Parameters:
    - data (pandas.DataFrame): The dataset containing the subject data.
    - worker_id (int or str): The worker ID of the subject.

    Returns:
    - pandas.DataFrame: The data for the specified subject.
    """
    return data[data["subid"] == worker_id]


def instantiate_model(model, subject_data, subject_params):
    """
    Instantiate a model with the given parameters and subject data.

    Args:
        model: The model to be instantiated.
        subject_data: The data for the subject.
        subject_params: The parameters for the subject.

    Returns:
        The instantiated model.

    """
    exclude_cols = ["subid", "model", "loglik", "aic"]
    param_cols = [col for col in subject_params.columns if col not in exclude_cols]
    params = subject_params[param_cols].values[0]
    return model(*params, subject_data)


def fit_model_to_record_RPE_LL_and_uncertainty(model, subject_data, subject_params):
    """
    Fits the given model to the subject data and parameters and returns the reward prediction error (RPE),
    trial-by-trial log-likelihood, and uncertainties.

    Parameters:
    - model: The model to fit (type: object)
    - subject_data: The data of the subject (type: object)
    - subject_params: The parameters of the subject (type: object)

    Returns:
    - rpe: The reward prediction error (type: object)
    - trial_by_trial_loglik: The trial-by-trial log-likelihood (type: object)
    - uncertainties: The uncertainties (type: object)
    """
    curr_model = instantiate_model(model, subject_data, subject_params)
    curr_model.fit()
    return curr_model.rpe, curr_model.trial_by_trial_loglik, curr_model.uncertainties


def combine_RPE_LL_and_uncertainty_with_sub_strat_data(
    sub_strat_data, rpe, ll, uncertainty
):
    """
    Combines the RPE (Reward Prediction Error), trial-by-trial log-likelihood, and uncertainty
    with the given sub_strat_data dictionary.

    Parameters:
    - sub_strat_data (dict): The dictionary containing sub-stratification data.
    - rpe (float): The reward prediction error value.
    - ll (float): The trial-by-trial log-likelihood value.
    - uncertainty (float): The uncertainty value.

    Returns:
    - sub_strat_data (dict): The updated sub_strat_data dictionary with the RPE, trial-by-trial log-likelihood,
      and uncertainty values added.
    """

    sub_strat_data["rpe"] = rpe
    sub_strat_data["trial_by_trial_loglik"] = ll
    sub_strat_data["uncertainty"] = uncertainty
    return sub_strat_data


def single_subj_proc(model, strat_data, fit_data, worker_id):
    """
    Process a single subject's data using the given model.

    Args:
        model: The model used for processing the data.
        strat_data: The strategic data for all subjects.
        fit_data: The fit data for all subjects.
        worker_id: The ID of the worker for whom the data is being processed.

    Returns:
        The processed strategic data for the specified worker.
    """
    sub_strat_data = get_single_subject_data(strat_data, worker_id)
    sub_fit_data = get_single_subject_data(fit_data, worker_id)
    rpe, ll, uncertainty = fit_model_to_record_RPE_LL_and_uncertainty(
        model, sub_strat_data, sub_fit_data
    )
    sub_strat_data = combine_RPE_LL_and_uncertainty_with_sub_strat_data(
        sub_strat_data, rpe, ll, uncertainty
    )
    return sub_strat_data


def fit_model_to_all_subjects(model, strat_data, fit_data):
    """
    Fits the given model to all subjects in the provided data.

    Parameters:
    - model: The model to fit.
    - strat_data: The stratified data containing subject information.
    - fit_data: The data used for fitting the model.

    Returns:
    - A list of results from fitting the model to each subject.
    """

    worker_ids = strat_data["subid"].unique()
    args = [(model, strat_data, fit_data, worker_id) for worker_id in worker_ids]

    return [single_subj_proc(*arg) for arg in args]


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Filler description")
    parser.add_argument(
        "-f", "--fit_input_path", required=True, help="path to the fit data"
    )
    parser.add_argument(
        "-s", "--strat_input_path", required=True, help="path to the strat data"
    ),
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Output path for where the new strat data will be saved",
    )
    args = parser.parse_args()
    fit_input_path = args.fit_input_path
    strat_input_path = args.strat_input_path
    output_path = args.output_path
    assert os.path.exists(fit_input_path), "fit input path does not exist!"
    assert os.path.exists(strat_input_path), "strat input path does not exist!"
    assert os.path.exists(output_path), "Output path does not exist!"

    fit_data = load_data(fit_input_path)
    strat_data = load_data(strat_input_path)

    # model name should be extracted from the fit_input_path the first part of the filename is the modelname
    model_name = fit_input_path.split("/")[-1].split("_")[0]

    model_dict = MODELS

    assert model_name.lower() in model_dict.keys(), "Model not recognized!"

    model = model_dict[model_name.lower()][0]

    strat_data_with_rpe = fit_model_to_all_subjects(model, strat_data, fit_data)
    # turn strat_data back into a dataframe
    strat_data_with_rpe = pd.concat(strat_data_with_rpe)
    output_filename = f"{model_name}_strat_data_rpe.csv"
    output_path = os.path.join(output_path, output_filename)
    strat_data_with_rpe.to_csv(output_path, index=False)
    print(f"Finished in {time.time() - t0} seconds")


if __name__ == "__main__":
    main()
