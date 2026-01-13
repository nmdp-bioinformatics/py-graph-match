from collections.abc import Iterator
from typing import List, Tuple, Set, Iterable, Dict
from typing import Sequence
from grma.match.hla_match import batch_locuses_match_between_genos

from ..utilities.utils import geno_to_int, gl_string_to_hash
import ast
from bidict import bidict
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from grma.match.graph_wrapper import Graph
from grma.utilities.geno_representation import HashableArray, ClassMinusOne
from grma.utilities.utils import (
    donor_mismatch_format,
    drop_less_than_7_matches,
    check_similarity,
    print_time,
)

DONORS_DB: pd.DataFrame = pd.DataFrame()
ZEROS: HashableArray = HashableArray([0])


def set_database(donors_db: pd.DataFrame = pd.DataFrame()):
    """
    Set a database for a search.
    Use this function before the matching if you wish to add fields for the result df.
    """
    global DONORS_DB
    DONORS_DB = donors_db


def _init_results_df(donors_info):
    """Initialize matching donors' df"""
    global DONORS_DB
    fields_in_results = {
        "Patient_ID": [],
        "Donor_ID": [],
        "Matching_Probability": [],
        "Match_Probability": [],
        "Permissive/Non-Permissive": [],
        # "Match_Between_Most_Commons": [],
    }

    # Define the mismatch types and their order
    mismatch_types = ["", "HvG", "GvH"]
    for t in mismatch_types:
        for i in range(4):  # 0 to 3 mismatches
            name = (
                f"chance_for_{i}_{t}_Mismatches" if t else f"chance_for_{i}_Mismatches"
            )
            fields_in_results[name] = []
        # Add "more than 3" for each mismatch type
        name = f"chance_for_3+_{t}_Mismatches" if t else f"chance_for_3+_Mismatches"
        fields_in_results[name] = []

    donors_db_fields = DONORS_DB.columns.values.tolist()
    for di in donors_info:
        if di in donors_db_fields:
            fields_in_results[di] = []
    return pd.DataFrame(fields_in_results)


# def locuses_match_between_genos(geno_pat, geno_don):
#     matches = []
#     total_gvh = 0
#     total_hvg = 0
#
#     for i in range(0, len(geno_pat), 2):
#         a1, b1 = geno_pat[i], geno_pat[i + 1]
#         a2, b2 = geno_don[i], geno_don[i + 1]
#
#         P = {x for x in (a1, b1) if x not in (None, 0)}
#         D = {x for x in (a2, b2) if x not in (None, 0)}
#
#         gvh_i = len(P - D)  # patient has, donor lacks
#         hvg_i = len(D - P)  # donor has, patient lacks
#
#         total_gvh += gvh_i
#         total_hvg += hvg_i
#
#         mismatch_i = max(gvh_i, hvg_i)  # table's #Max
#         matches.append(mismatch_i)
#
#     return matches, total_gvh, total_hvg


class DonorsMatching(object):
    """DonorsMatching class is in charge of the matching process"""

    __slots__ = (
        "_graph",
        "_patients_graph",
        "_genotype_candidates",
        "patients",
        "verbose",
        "bidirectional_dict",
        "_donor_neighbors_cache",
    )

    def __init__(self, graph: Graph, verbose: bool = False, bdict: bidict = None):
        self._graph: Graph = graph
        self._patients_graph: nx.DiGraph = nx.DiGraph()
        self._genotype_candidates: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}
        self.patients: Dict[int, Dict[int, HashableArray]] = {}
        self.verbose = verbose
        self.bidirectional_dict = bdict
        self._donor_neighbors_cache = {}

    def get_most_common_genotype(self, donor_id):
        """Takes a donor ID and return his/her most common genotype."""
        don_geno = []
        geno_max_prob = 0
        for geno in self._graph.neighbors(donor_id):
            if geno[1] > geno_max_prob:
                geno_max_prob = geno[1]
                don_geno = geno[0]

        return don_geno

    def print_most_common_genotype(self, don_id: int, pat_geno: Sequence[int]) -> str:
        """Takes a donor ID and a genotype.
        Returns the mismatch format of the most common genotype of the donor."""
        don_geno = []
        geno_max_prob = 0
        for geno in self._graph.neighbors(don_id):
            if geno[1] > geno_max_prob:
                geno_max_prob = geno[1]
                don_geno = geno[0]

        return donor_mismatch_format(don_geno, pat_geno)

    def probability_to_allele(
        self, don_id: int, pat_geno: Sequence[int]
    ) -> List[float]:
        """Takes a donor ID and a genotype.
        Returns the probability of match for each allele"""
        probs = [0 for _ in range(len(pat_geno))]

        for i, allele in enumerate(pat_geno):
            p = 0
            for don_geno, don_weight in self._graph.neighbors(don_id):
                if allele in don_geno:
                    p += don_weight
            probs[i] = int(round(p * 100))

        return probs

    def __find_genotype_candidates_from_subclass(self, sub: int) -> np.ndarray:
        """Takes an integer subclass.
        Returns the genotypes which are connected to it in the graph"""
        return self._graph.neighbors_2nd(sub)

    def __find_genotype_candidates_from_class(
        self, clss: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Takes an integer subclass.
        Returns the genotypes (ids and values) which are connected to it in the graph"""
        return self._graph.class_neighbors(
            clss, Len=len(next(iter(next(iter(self.patients.values())).values())))
        )

    def __find_donor_from_geno(self, geno_id: int) -> Sequence[int]:
        """Gets the LOL ID of a genotype.
        Return its neighbors - all the donors that has this genotype."""
        ids, _ = zip(*self._graph.neighbors(geno_id, search_lol_id=True))
        return list(ids)

    def __add_matched_genos_to_graph(
        self,
        genos: Iterator,
        genotypes_ids: np.ndarray,
        genotypes_values: np.ndarray,
        allele_range_to_check: np.ndarray,
        matched_alleles: int,
    ):
        for geno in genos:
            # check similarity between geno and all the candidates
            similarities = check_similarity(
                geno.np(), genotypes_values, allele_range_to_check, matched_alleles
            )

            candidates_to_iterate = drop_less_than_7_matches(
                genotypes_ids, similarities
            )

            for geno_candidate_id, similarity in candidates_to_iterate:
                # iterate over all the patients with the genotype
                for patient_id in self._patients_graph.neighbors(geno):
                    # patient's geno index (the number of the geno in the imputation file)
                    geno_num = self._patients_graph[geno][patient_id]["geno_num"]
                    probability = self._patients_graph[geno][patient_id][
                        "probability"
                    ]  # patient's geno probability

                    # STUDY TEST CASE
                    # problem_node = 26529534
                    # problem_patient = 26477347
                    # if patient_id == problem_node and geno_candidate_id == problem_patient:
                    #     x = 1
                    # FINISH STUDY TEST CASE

                    # add the genotype id as a neighbor to the patient
                    if geno_candidate_id in self._genotype_candidates[patient_id]:
                        self._genotype_candidates[patient_id][geno_candidate_id][
                            geno_num
                        ] = (probability, similarity)
                    else:
                        self._genotype_candidates[patient_id][geno_candidate_id] = {
                            geno_num: (probability, similarity)
                        }

                    """
                    if geno_candidate_id in self._patients_graph.adj[patient_id]:
                        self._patients_graph[patient_id][geno_candidate_id]['weight'][geno_num] = [probability,
                                                                                                   similarity]
                    else:
                        self._patients_graph.add_edge(patient_id, geno_candidate_id,
                                                      weight={geno_num: [probability, similarity]})
                    """

    def __classes_and_subclasses_from_genotype(self, genotype):
        subclasses = []
        ALLELES_IN_CLASS_I = -2 * int(-len(genotype) / 4 - 0.5)
        ALLELES_IN_CLASS_II = len(genotype) - ALLELES_IN_CLASS_I
        classes = [
            (tuple(genotype[:ALLELES_IN_CLASS_I]), 0),
            (tuple(genotype[ALLELES_IN_CLASS_I:]), 1),
        ]
        num_of_alleles_in_class = [ALLELES_IN_CLASS_I, ALLELES_IN_CLASS_II]

        int_classes = []
        for clss in classes:
            hash_clss = gl_string_to_hash(str(clss[0])) % 1000000000 + 1000000000
            int_classes.append((hash_clss, clss[1]))

        for clss in int_classes:
            self._patients_graph.add_edge(clss[0], genotype)

        # class one is considered as 0.
        # class two is considered as 1.
        class_options = [0, 1]
        for class_num in class_options:
            for k in range(0, num_of_alleles_in_class[class_num]):
                # set the missing allele to always be the second allele in the locus
                if k % 2 == 0:
                    subclass = tuple(
                        classes[class_num][0][0:k]
                        + (0,)
                        + classes[class_num][0][k + 1 :]
                    )
                else:
                    subclass = tuple(
                        classes[class_num][0][0 : k - 1]
                        + (0,)
                        + classes[class_num][0][k - 1 : k]
                        + classes[class_num][0][k + 1 :]
                    )
                hash_subclass = (
                    gl_string_to_hash(str(subclass)) % 1000000000 + 2000000000
                )

                # missing allele number is the index of the first allele of the locus the missing allele belongs to.
                # Could be [0, 2, 4, 6, 8]
                missing_allele_num = ALLELES_IN_CLASS_I * class_num + 2 * (k // 2)
                subclass = ClassMinusOne(
                    subclass=hash_subclass,
                    class_num=class_num,
                    allele_num=missing_allele_num,
                )

                # add subclass -> genotype edge to patients graph
                subclasses.append(subclass)

                self._patients_graph.add_edge(subclass, genotype)

        return int_classes, subclasses

    def create_patients_graph(self, f_patients: str):
        """
        create patients graph. \n
        *takes in consideration that grimm outputs for each patient different genotypes*
        """
        self._patients_graph: nx.DiGraph = nx.DiGraph()
        prob_dict: dict = {}  # {geno: [i, prob]}
        total_prob: float = 0
        last_patient: int = -1
        # subclasses: list[ClassMinusOne] = []
        subclasses_by_patient: Dict[int, Set] = {}
        classes_by_patient: Dict[int, Set] = {}

        for line in open(f_patients).readlines():
            # retrieve all line's parameters
            line_values = line.strip().split(",")
            patient_id, geno, prob, index = line_values
            patient_id = -1 * int(patient_id)
            index = int(index)
            prob = float(prob)

            encoded = []
            for locus in geno.split("^"):
                parts = locus.split("+")
                aL = parts[0].strip() if len(parts) > 0 else "N"
                aR = parts[1].strip() if len(parts) > 1 else "N"

                # Classify left allele
                if "N" in aL:
                    tL, vL = "missing", 0
                elif aL in self.bidirectional_dict:
                    tL, vL = "known", self.bidirectional_dict[aL]
                else:
                    tL, vL = "unknown", None

                # Classify right allele
                if "N" in aR:
                    tR, vR = "missing", 0
                elif aR in self.bidirectional_dict:
                    tR, vR = "known", self.bidirectional_dict[aR]
                else:
                    tR, vR = "unknown", None

                # Decide per-locus encoding (two outputs per locus, in order)
                if tL == "unknown" and tR == "unknown":
                    if aL == aR:
                        encoded.extend(
                            [1, 1]
                        )  # same unknown twice (homozygous unknown)
                    else:
                        encoded.extend([1, 2])  # two different unknowns
                elif tL == "unknown" and tR != "unknown":
                    encoded.extend([1, vR])  # left unknown, right known/missing
                elif tL != "unknown" and tR == "unknown":
                    encoded.extend([vL, 1])  # left known/missing, right unknown
                else:
                    encoded.extend([vL, vR])  # both known/missing

            geno = HashableArray(encoded)
            if patient_id in self.patients.keys():
                self.patients[patient_id][index] = geno
            else:
                self.patients[patient_id] = {index: geno}
            if index == 0:
                # set normalized probabilities
                for HLA, probability in prob_dict.items():
                    self._patients_graph.edges[HLA, last_patient]["probability"] = (
                        probability / total_prob
                    )

                # initialize parameters
                prob_dict = {}
                total_prob = 0
                self._genotype_candidates[patient_id] = {}
                last_patient = patient_id

                subclasses_by_patient[patient_id] = set()
                classes_by_patient[patient_id] = set()

            # add probabilities to probability dict
            total_prob += prob
            if geno not in prob_dict:
                prob_dict[geno] = prob
            else:
                prob_dict[geno] += prob

            # add genotype->ID edge
            self._patients_graph.add_edge(
                geno, patient_id, probability=0, geno_num=index
            )

            # add subclasses alleles
            classes, subclasses = self.__classes_and_subclasses_from_genotype(geno)

            subclasses_by_patient[patient_id] = subclasses_by_patient[patient_id].union(
                subclasses
            )
            classes_by_patient[patient_id] = classes_by_patient[patient_id].union(
                classes
            )

        # set normalized probabilities to the last patient in the file
        for HLA, probability in prob_dict.items():
            self._patients_graph.edges[HLA, last_patient]["probability"] = (
                probability / total_prob
            )

        # return subclasses_by_patient
        return subclasses_by_patient, classes_by_patient

    def find_geno_candidates_by_subclasses(self, subclasses):
        for subclass in tqdm(
            subclasses,
            desc="finding subclasses matching candidates",
            disable=not self.verbose,
        ):
            if self._graph.in_nodes(subclass.subclass):
                patient_genos = self._patients_graph.neighbors(
                    subclass
                )  # The patient's genotypes which might be match
                (
                    genotypes_id,
                    genotypes_value,
                ) = self.__find_genotype_candidates_from_subclass(subclass.subclass)

                len_geno = len(genotypes_value[0])
                ALLELES_IN_CLASS_I = -2 * int(-len_geno / 4 - 0.5)
                ALLELES_IN_CLASS_II = len_geno - ALLELES_IN_CLASS_I
                # Checks only the locuses that are not certain to match
                if subclass.class_num == 0:
                    allele_range_to_check = np.array(
                        [x for x in range(ALLELES_IN_CLASS_I, len_geno, 2)]
                        + [subclass.allele_num],
                        dtype=np.uint8,
                    )
                else:
                    allele_range_to_check = np.array(
                        [x for x in range(0, ALLELES_IN_CLASS_I, 2)]
                        + [subclass.allele_num],
                        dtype=np.uint8,
                    )

                # number of alleles that already match due to match in subclass
                matched_alleles: int = (
                    ALLELES_IN_CLASS_II
                    if subclass.class_num == 1
                    else ALLELES_IN_CLASS_I
                ) - 2

                # Compares the candidate to the patient's genotypes, and adds the match geno candidates to the graph.
                self.__add_matched_genos_to_graph(
                    patient_genos,
                    genotypes_id,
                    genotypes_value,
                    allele_range_to_check,
                    matched_alleles,
                )

    def find_geno_candidates_by_classes(self, classes):
        for clss in tqdm(
            classes,
            desc="finding classes matching candidates",
            disable=not self.verbose,
        ):
            if self._graph.in_nodes(clss[0]):
                patient_genos = self._patients_graph.neighbors(
                    clss[0]
                )  # The patient's genotypes which might be match
                (
                    genotypes_ids,
                    genotypes_values,
                ) = self.__find_genotype_candidates_from_class(clss)

                # Checks only the locuses that are not certain to match (the locuses of the other class)
                # Class I appearances: 3 locuses = 6 alleles = 23/24 digits
                # Class II appearances: 2 locuses = 4 alleles = 15/16 digits
                num_alleles = len(
                    next(iter(next(iter(self.patients.values())).values()))
                )
                if clss[1] == 0:
                    allele_range_to_check = np.array(
                        [
                            x
                            for x in range(
                                num_alleles // 2 + (num_alleles // 2 & 1),
                                num_alleles,
                                2,
                            )
                        ],
                        dtype=np.uint8,
                    )
                    matched_alleles: int = num_alleles // 2 + (num_alleles // 2 & 1)

                else:
                    allele_range_to_check = np.array(
                        [
                            x
                            for x in range(
                                0, num_alleles // 2 + (num_alleles // 2 & 1), 2
                            )
                        ],
                        dtype=np.uint8,
                    )
                    matched_alleles: int = num_alleles // 2 - (num_alleles // 2 & 1)

                # Compares the candidate to the patient's genotypes, and adds the match geno candidates to the graph.
                self.__add_matched_genos_to_graph(
                    patient_genos,
                    genotypes_ids,
                    genotypes_values,
                    allele_range_to_check,
                    matched_alleles,
                )

                # Send the class and the genotypes of the patients that the class belong to
                # self.__add_class_candidates(clss, self._patients_graph.neighbors(clss))

    def find_geno_candidates_by_genotypes(self, patient_id: int):
        genos = self._patients_graph.predecessors(patient_id)

        for geno in genos:
            # if patient_id in self._patients_graph[geno]:
            #     print("Processing geno:", geno)
            #     print("Processing patient_id:", patient_id)
            # print("Dictionary contents:", self._patients_graph[geno][patient_id])

            # if "geno_num" in self._patients_graph[geno][patient_id]:
            #     print("Patient ID:", patient_id, "has 'geno_num'")
            geno_num = self._patients_graph[geno][patient_id][
                "geno_num"
            ]  # patient's geno index
            probability = self._patients_graph[geno][patient_id][
                "probability"
            ]  # patient's geno probability

            geno_id = self._graph.get_node_id(geno)
            if not geno_id:
                continue

            # This has to be a new edge because this is the first level (searching by genos),
            # and each patient connects only to their own genos, so we wouldn't override the weight dict.
            # self._patients_graph.add_edge(patient_id, geno_id, weight={geno_num: [probability, 10]})
            self._genotype_candidates[patient_id][geno_id] = {
                geno_num: (probability, len(geno))
            }
            # else:
            #     print(f"Missing 'geno_num' for patient_id: {patient_id}")
            #     print("geno:", geno)
            #     print("patient_id:", patient_id)
            # else:
            #     print(f"Patient ID {patient_id} not found in self._patients_graph[geno]")
            #     print("geno:", geno)
            #     print("patient_id:", patient_id)

        """
        genos = self._patients_graph.predecessors(patient_id)
        for geno in genos:
            print(f"Processing geno: {geno}, patient_id: {patient_id}")

            # geno_num = self._patients_graph[geno][patient_id]["geno_num"]  # patient's geno index
            probability = self._patients_graph[geno][patient_id]["probability"]  # patient's geno probability

            int_geno = tuple_geno_to_int(geno)
            geno_id = self._graph.get_node_id(int_geno)
            if not geno_id:
                continue

            # This has to be a new edge, because this is the first level (searching by genos),
            # and each patient connect only to his own genos, so we wouldn't override the weight dict.
            # self._patients_graph.add_edge(patient_id, geno_id, weight={geno_num: [probability, 10]}
            self._genotype_candidates[patient_id][geno_id] = [(geno_num, probability, 10)]

        """

    def score_matches(
        self,
        mismatch: int,
        results_df: pd.DataFrame,
        donors_info: Iterable[str],
        patient: int,
        threshold: float,
        cutoff: int,
        matched: Set[int],
    ) -> Tuple[Set[int], int, pd.DataFrame]:
        """
        Given a number of mismatches and a patient, this function will return a tuple
        of all matching donors found in the data with the specific number of mismatches,
        sorted by their probability for a match.

        :param mismatch: number of mismatch to search. could be 0, 1, 2, 3.
        :param results_df: A df storing the matching results.
        :param donors_info: a list of fields from Database to add to the matching results.
        :param patient: patient ID.
        :param threshold: Minimal score value for a valid match. default is 0.1.
        :param cutoff: Maximum number of matches to return. default is 50.
        :param matched: A set of donors ID that have already matched for this patient.
        :return: a tuple of (matched set, count of matches, updated results DataFrame).
        """
        if len(matched) >= cutoff:
            return matched, 0, results_df
        num_alleles = len(next(iter(self.patients[patient].values())))

        # a loop that set the scores for all the matching candidates.
        patient_scores = {}
        # for hla_id in self._patients_graph.neighbors(patient):
        for hla_id, genotype_matches in self._genotype_candidates[patient].items():
            for prob, matches in genotype_matches.values():
                # match_info = (probability of patient's genotype, number of matches to patient's genotype)
                if matches != num_alleles - mismatch:
                    continue

                # add the probabilities multiplication of the patient and all the donors that has this genotype
                # to their matching probabilities.
                for donor in self.__find_donor_from_geno(hla_id):
                    donor_prob = self._graph.get_edge_data(
                        node1=hla_id, node2=donor, node1_id=True
                    )
                    if donor in patient_scores:
                        patient_scores[donor][0] += prob * donor_prob
                        if donor_prob > patient_scores[donor][2]:
                            patient_scores[donor][1:] = [hla_id, donor_prob]
                    else:
                        patient_scores[donor] = [prob * donor_prob, hla_id, donor_prob]

        ids_scores = []
        count_matches = 0

        # sort matching according to their probability
        for donor in patient_scores.keys():
            # do not count or match to an already matched donors.
            if donor in matched or patient_scores[donor][0] < threshold:
                continue

            count_matches += 1
            ids_scores.append((donor, patient_scores[donor][0]))

        ids_scores.sort(reverse=True, key=lambda x: x[1])

        add_donors = {col: [] for col in results_df.columns.values.tolist()}

        # write matching donors to results.
        for donor, score in ids_scores:
            if len(matched) >= cutoff:
                break
            matched.add(donor)
            # Compute chances for gvh, hvg, and mis
            pats = self.patients[patient]
            chances = {
                "gvh": [0, 0, 0, 0, 0],
                "hvg": [0, 0, 0, 0, 0],
                "mis": [0, 0, 0, 0, 0],
            }

            for index, pat in pats.items():
                pat_prob = self._patients_graph[pat][patient]["probability"]

                # --- cache donor genotype neighbors once per donor ---
                if not hasattr(self, "_donor_neighbors_cache"):
                    self._donor_neighbors_cache = {}
                donor_neighbors = self._donor_neighbors_cache.setdefault(
                    donor, list(self._graph.neighbors(donor))
                )

                if not donor_neighbors:
                    continue

                # Build 2D array of donor genotypes (rows) and 1D array of their probs
                # Assumes each geno[0] is a fixed-length tuple/list of ints (same len as pat.np()).
                don_genos = np.array(
                    [geno[0] for geno in donor_neighbors], dtype=np.uint32
                )
                don_probs = np.array([geno[1] for geno in donor_neighbors], dtype=float)

                # Ensure C-contiguous (should already be, but just to be safe)
                if not don_genos.flags.c_contiguous:
                    don_genos = np.ascontiguousarray(don_genos, dtype=np.uint32)

                pat_np = np.asarray(
                    pat.np(), dtype=np.uint32
                )  # no copy if already uint32 + contiguous

                # One Cython call for all donor genos
                # out[:,0]=mis, out[:,1]=gvh, out[:,2]=hvg
                out = batch_locuses_match_between_genos(pat_np, don_genos)

                # Accumulate chances
                # (same logic as before; clamp at 4 for "3+")
                for j in range(out.shape[0]):
                    mis = int(out[j, 0])
                    gvh = int(out[j, 1])
                    hvg = int(out[j, 2])
                    w = pat_prob * don_probs[j]
                    chances["gvh"][min(gvh, 4)] += w
                    chances["hvg"][min(hvg, 4)] += w
                    chances["mis"][min(mis, 4)] += w

            # Compute allele probability
            allele_prob = self.probability_to_allele(
                don_id=donor, pat_geno=self.patients[patient][0].np()
            )

            # Append donor data to add_donors
            add_donors["Patient_ID"].append(-1 * patient)
            add_donors["Donor_ID"].append(-1 * donor)
            add_donors["Match_Probability"].append(allele_prob)
            # add_donors["Match_Between_Most_Commons"].append(compare_commons)
            add_donors["Matching_Probability"].append(score)
            for t in ("GvH", "HvG", ""):
                for i in range(5):
                    key = f"chance_for_{'3+' if i == 4 else i}_{t + '_Mismatches' if t else 'Mismatches'}"
                    add_donors[key].append(chances[t.lower() if t else "mis"][i])

            add_donors["Permissive/Non-Permissive"].append("-")
            # TODO: add permissiveness algorithm

            # add the other given fields to the results
            for field in donors_info:
                add_donors[field].append(DONORS_DB.loc[donor, field])

        results_df = pd.concat(
            [results_df, pd.DataFrame(add_donors)], ignore_index=True
        )

        if self.verbose:
            print_time(f"({mismatch} MMs) Found {count_matches} matches")

        return matched, count_matches, results_df

    @property
    def patients_graph(self):
        return self._patients_graph
