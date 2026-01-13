import os
import sys
import time
import json
from os import PathLike
from typing import Iterable, Union, Dict
import pickle
import pandas as pd
from grim import grim
import csv
import ast
from bidict import bidict
import re

from grma.match import Graph as MatchingGraph
from grma.match.donors_matching import DonorsMatching, _init_results_df
from grma.match.graph_wrapper import Graph
from grma.utilities.utils import print_time, donor_mismatch_format

GRIM_DEFAULT_OUTPUT_PATH = "./output/don.pmug"
GRIM_RESULT_DIR_FIELD = "imuptation_out_path"
GRIM_RESULT_GENO_FILE_FIELD = "imputation_out_umug_freq_filename"


def run_grim(config_file_path=""):
    """ "
    This function applies grim imputation with the default/user's configuration file.

    :param config_file_path: A path to configuration file for grim imputation. default is grim default config.
    :return: The path to the imputation output file.
    """
    if not config_file_path:
        grim.graph_freqs()
        grim.impute()
        return GRIM_DEFAULT_OUTPUT_PATH

    if not (os.path.exists(config_file_path) and os.path.isfile(config_file_path)):
        raise FileExistsError(f"Can't find configuration file in {config_file_path}.")

    with open(config_file_path, "r") as f:
        conf = json.load(f)

    # Finds the path of the output of grim for
    dir_path = conf[GRIM_RESULT_DIR_FIELD]
    file_name = conf[GRIM_RESULT_GENO_FILE_FIELD]
    path = os.path.join(dir_path, file_name)

    grim.graph_freqs(config_file_path)
    grim.impute(config_file_path)

    return path


def search_in_levels(
    patient_id: int,
    g_m: DonorsMatching,
    donors_info: Iterable,
    threshold: float,
    cutoff: int,
    classes: Iterable,
    subclasses: Iterable,
):
    """ "
    This function gets the information about the patient and other settings, and search for donors in levels.
    First, it will search only for 10 matches donors (compare only genotypes). If there aren't enough donors,
    the function will search for 9 matches donors (compare classes), and then 8-7 matches (subclasses)

    :param patient_id: The id of the patient we want to search his/her matches.
    :param g_m: The patients graph with the info of the current patient (and maybe other patients).
    :param donors_info: An iterable of fields from the database to include in the results. default is None.
    :param threshold: Minimal score value for a valid match. default is 0.1.
    :param cutoff: Maximum number of matches to return. default is 50.
    :param classes: An iterable with the all the possible classes of the patient.
    :param subclasses: An iterable with the all the possible subclasses of the patient.
    :return: A pandas.DataFrame with the matches for this patient.
    """

    matched = set()  # set of donors ID that have already matched for this patient
    results_df = _init_results_df(
        donors_info
    )  # initialize the df according to the given fields
    # print(f"Before find_geno_candidates_by_genotypes: patient_id={patient_id}")
    # We can give to this function the genotypes instead
    g_m.find_geno_candidates_by_genotypes(patient_id)
    # print(f"After find_geno_candidates_by_genotypes")

    matched, count, results_df = g_m.score_matches(
        0, results_df, donors_info, patient_id, threshold, cutoff, matched
    )

    if len(matched) >= cutoff:
        return results_df

    g_m.find_geno_candidates_by_classes(classes)
    matched, count, results_df = g_m.score_matches(
        1, results_df, donors_info, patient_id, threshold, cutoff, matched
    )

    if len(matched) >= cutoff:
        return results_df

    g_m.find_geno_candidates_by_subclasses(subclasses)

    # loop over possible mismatches: 2, 3.
    for mismatches in range(2, 4):
        matched, count, results_df = g_m.score_matches(
            mismatches, results_df, donors_info, patient_id, threshold, cutoff, matched
        )

    return results_df


def find_matches(
    imputation_filename: Union[str, PathLike],
    match_graph: Graph,
    search_id: int = 1,
    donors_info: Iterable[str] = [],
    threshold: float = 0.1,
    cutoff: int = 100,
    verbose: bool = False,
    save_to_csv: bool = False,
    calculate_time: bool = False,
    output_dir: str = "output",
    bdict: bidict = None,
):
    """
    The main function responsible for performing the matching.
    Note: for each patient, if a donor has been found as a
    match in an early stage (0, 1, or 2 mm), he will not be searched as a match for the further mismatches.

    :param imputation_filename: Path to the output file of the imputation made by grim.
    :param match_graph: A Graph object from grma.match
    :param search_id: An integer identification of the search. default is 0.
    :param donors_info: An iterable of fields from the database to include in the results. default is None.
    :param threshold: Minimal score value for a valid match. default is 0.1.
    :param cutoff: Maximum number of matches to return. default is 50.
    :param verbose: A boolean flag for whether to print the documentation. default is False
    :param save_to_csv: A boolean flag for whether to save the matching results into a csv file. default is False.
    :param calculate_time: A boolean flag for whether to return the matching time for patient. default is False.
    If one wishes to save the results to csv files, a directory named 'Matching_Results_{searchId}' will be created in
    the working directory. If a directory by this name was already created, an error will be raised.
    Note: saving a pandas into a csv might take a couple of seconds
    :return: A dictionary that maps each patient to its matching results formatted as a pandas.DataFrame
    """
    if save_to_csv:
        if verbose:
            print(f"Creating {output_dir}/search_{search_id}")
        os.makedirs(f"{output_dir}/search_{search_id}", exist_ok=True)

    if verbose:
        print_time("Start graph matching")

    g_m = DonorsMatching(match_graph, verbose=verbose, bdict=bdict)

    # create patients graph and find all candidates
    start_build_graph = time.time()
    subclasses_by_patient, classes_by_patient = g_m.create_patients_graph(
        imputation_filename
    )

    # wierd_edges = {}
    # for patient in g_m.patients:
    #     genos = list(g_m.patients_graph.predecessors(patient))
    #     for geno in genos:
    #         edge = g_m.patients_graph[geno][patient]
    #         if "geno_num" not in edge or "probability" not in edge:
    #             wierd_edges[patient] = geno
    #
    # print(len(wierd_edges))

    # print("------------------------------Fuck My Life-----------------------------------------------------------------")
    # for patient, genos in g_m.patients.items():
    #     procces = set(g_m.patients_graph.predecessors(patient))
    #     if len(procces) != 1:
    #         print(patient)
    #     patient_predecessors = [p for p in procces if p]
    #     patient_predecessors = list(patient_predecessors[0])
    #
    #     if patient_predecessors != genos:
    #         print(f'{patient}: is not good')
    #
    # patient_ids = []
    #
    # with open(imputation_filename, 'r') as file:
    #     reader = csv.reader(file, delimiter=',')
    #     for row in reader:
    #         patient_id = row[0].split(',')[0]
    #         patient_ids.append(patient_id)
    # print(len(patient_ids))
    patients = list(g_m.patients.keys())
    # print(len(patients))
    # if patients == patient_ids:
    #     print("--------------------TRUE--------------------------")

    if verbose:
        print_time("Created patients graph")
    end_build_graph = time.time()

    # the returned dictionary. {patient ID: pd.DataFrame(matches + features)}
    patients_results = {patient: None for patient in patients}
    with open(imputation_filename, "r") as f:
        line = f.readline().strip()
        text = line.split(",")[1]
        raw_loci = re.findall(r"\b([A-Za-z0-9]+)\*", text)
        loci = []
        for locus in raw_loci:
            if locus in ("DRB3", "DRB4", "DRB5", "DRBX"):
                locus = "DRB3/4/5/X"
            loci.append(locus)
        loci = list(dict.fromkeys(loci))
    if patients:
        avg_build_time = (end_build_graph - start_build_graph) / len(patients)
    else:
        avg_build_time = 0

    times = []
    for patient in patients:
        # print("\n","Patient", patient, "Verbose", verbose)
        # For each patient we search matches in the donor graph.
        # First we will look for perfect matches - only genotypes, then 9 matches - classes,
        # and then 7-8 matches - subclasses.
        if verbose:
            print_time(f"Searching matches for {patient}")

        start = time.time()

        subclasses = subclasses_by_patient[patient]
        classes = classes_by_patient[patient]
        results_df = search_in_levels(
            patient, g_m, donors_info, threshold, cutoff, classes, subclasses
        )

        match_Probability = results_df["Match_Probability"]
        # match_Between_Most_Commons = results_df["Match_Between_Most_Commons"]
        Permissive = results_df["Permissive/Non-Permissive"]
        # df_new = results_df.drop(columns=['Match_Probability', 'Match_Between_Most_Commons', 'Permissive/Non-Permissive']).copy()
        df_new = results_df.drop(
            columns=["Match_Probability", "Permissive/Non-Permissive"]
        ).copy()

        for idx, row in enumerate(match_Probability):
            k = 0
            for locus in loci:
                for i in [1, 2]:
                    df_new.loc[idx, f"Match_Probability_{locus}_{i}"] = row[k]
                    k += 1
        df_new["Permissive/Non-Permissive"] = Permissive
        # for idx, row in enumerate(match_Between_Most_Commons):
        #     for l, locus in enumerate(loci):
        #         df_new.loc[idx, f"Match_Between_Most_Commons_{locus}"] = row[l]
        results_df = df_new.copy()

        # Normalize mismatch probability columns
        # for cols in [
        #     ['chance_for_0_GvH_Mismatches', 'chance_for_1_GvH_Mismatches', 'chance_for_2_GvH_Mismatches', 'chance_for_3_GvH_Mismatches', 'chance_for_3+_GvH_Mismatches'],
        #     ['chance_for_0_HvG_Mismatches', 'chance_for_1_HvG_Mismatches', 'chance_for_2_HvG_Mismatches', 'chance_for_3_HvG_Mismatches', 'chance_for_3+_HvG_Mismatches'],
        #     ['chance_for_0_Mismatches', 'chance_for_1_Mismatches', 'chance_for_2_Mismatches', 'chance_for_3_Mismatches', 'chance_for_3+_Mismatches']
        # ]:
        #     prob_sum = results_df[cols].sum(axis=1)
        #     for col in cols:
        #         results_df[col] = results_df[col].div(prob_sum).fillna(0.0).round(2)

        end = time.time()
        patient_time = end - start + avg_build_time
        times.append(end - start)
        if calculate_time:
            patients_results[patient] = (results_df, patient_time)
        else:
            patients_results[patient] = results_df
        patient = -1 * patient
        if save_to_csv:
            # save results to csv
            results_df.to_csv(
                os.path.join(
                    f"{output_dir}/search_{search_id}", f"Patient_{patient}.csv"
                ),
                index=True,
                # float_format="%.2f",
            )
            if verbose:
                print_time(
                    f"Saved Matching results for {patient} in "
                    f"{os.path.join(f'{output_dir}/search_{search_id}', f'Patient_{patient}.csv')}"
                )
    with open("times_grma.json", "w") as f:
        json.dump(times, f, indent=4)
    return patients_results


def matching(
    match_graph: MatchingGraph,
    grim_config_file="",
    save_imputation: Union[bool, str, PathLike] = False,
    donors_info: Union[Iterable[str], None] = None,
    search_id: int = 0,
    threshold: float = 0.1,
    cutoff: int = 100,
    verbose: bool = False,
    save_to_csv: bool = False,
    output_dir="output",
    bdict: bidict = None,
):
    """
    A function that performs the patients imputation with the matching.
    The imputation is performed with GRIM algorithm.

    :param match_graph: A Graph object from grma.match
    :param grim_config_file: A path to configuration file for grim imputation. default is grim default config.
    :param save_imputation: A flag for whether to save the imputation results. default is False.
    Accepts boolean/str/PathLike values - False will not save a file,
    True will save a file named 'imputation{searchId}.csv' in the working directory.
    str/PathLike will create and save the given file as long as the file's directory is valid.
    Otherwise, the file will not be saved. default is False.
    :param donors_info: An iterable of fields from the database to include in the results. default is None.
    :param search_id: An integer identification of the search. default is 0.
    :param threshold: Minimal score value for a valid match. default is 0.1.
    :param cutoff: Maximum number of matches to return. default is 50.
    :param verbose: A boolean flag for whether to print the documentation. default is False
    :param save_to_csv: A boolean flag for whether to save the matching results into a csv file. default is False.
    If one wishes to save the results to csv files, a directory named 'Matching_Results_{searchId}' will be created in
    the working directory. If a directory by this name was already created, an error will be raised.
    Note: saving a pandas into a csv might take a couple of seconds
    :return: A dictionary that maps each patient to its matching results formatted as a pandas.DataFrame
    """
    if donors_info is None:
        donors_info = []

    # disable output from grim
    if not verbose:
        sys.stdout = open(os.devnull, "w")

    imputation_path: str = run_grim(grim_config_file)

    # enable output if disabled
    if not verbose:
        sys.stdout = sys.__stdout__

    all_matches: Dict[int, pd.DataFrame] = find_matches(
        imputation_path,
        match_graph,
        search_id,
        donors_info,
        threshold,
        cutoff,
        verbose,
        save_to_csv,
        output_dir=output_dir,
        bdict=bdict,
    )

    return all_matches
