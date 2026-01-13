from grma.match import Graph, matching


def main():
    PATH_TO_DONORS_GRAPH = "output/donors_graph.pkl"
    PATH_CONFIG_FILE = "conf/minimal-configuration.json"

    # The donors' graph we built earlier
    donors_graph, allele_idx_bidict = Graph.from_pickle(PATH_TO_DONORS_GRAPH)

    # matching_results is a dict - {patient_id: the patient's result dataframe}
    matching_results = matching(
        donors_graph,
        PATH_CONFIG_FILE,
        search_id=1,
        donors_info=[],
        threshold=0.1,
        cutoff=100,
        save_to_csv=True,
        verbose=True,
        output_dir="results",
        bdict=allele_idx_bidict,
    )


if __name__ == "__main__":
    main()
