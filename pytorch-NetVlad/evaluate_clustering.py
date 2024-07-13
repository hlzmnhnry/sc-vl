import json
import faiss
import argparse
import numpy as np
import fourseasons as dataset

from os.path import *
from tqdm import tqdm

# tag for overview
stepsize = 100

# hard-coded how many clusters are (maximally) included
size_consideration = {
    "hard-clustering": 1,
    "soft-top-2": 2,
    "soft-top-3": 3,
    "soft-threshold-0.34": 2,
    "soft-threshold-0.26": 3
}

# recall@n values
n_values = [1, 5, 10, 20]

# hard-coded for experiments
height_levels = [300, 400, 500, 600]
my_methods = ["SameSizeKMeans", "KMeans", "GaussianMixture"]
cluster_nums = range(3, 6+1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate (semantic) clustering of DB", add_help=True)

    parser.add_argument("--query_features_file", type=str,
        help="Path to input file containing vpr query descriptors",
        default="query_features_val.npy")
    parser.add_argument("--db_features_file", type=str,
        help="Path to input file containing vpr db descriptors",
        default="database_features_val.npy")
    parser.add_argument("--assignment_files_directory", type=str,
        help="Path to file containing files from assign_clusters",
        default="../sem-clustering/results/")
    parser.add_argument("--type", type=str,
        help="Type of clustering", choices=["sem", "vpr"],
        default="sem")
    parser.add_argument("--result_tag", type=str,
        help="Suffix for result file",
        default="")

    args = parser.parse_args()

    # load descriptors from test with --storetest
    query_features = np.load(args.query_features_file)
    database_features = np.load(args.db_features_file)

    whole_test_set = dataset.get_whole_val_set()

    # for each query get those within threshold distance
    gt = whole_test_set.getPositives()

    results = {}

    for method in my_methods:

        print(method)

        for key in cluster_nums:

            print(f"\t'->{key}")

            # for each 'strategy' store max(n_values) 
            # (i.e. 20) predictions for each query
            predictions = {
                "hard-clustering": [],
                "soft-top-2": [],
                "soft-top-3": [],
                "soft-threshold-0.26": [],
                "soft-threshold-0.34": []
            }

            # keep track of all size reg. to individ. strategies
            search_space_sizes = {strategy: [] for strategy in predictions.keys()}

            for hl in height_levels:

                print(f"\t\t'->{hl}")

                references_cluster = json.load(open(join(args.assignment_files_directory,
                    f"clusters_{method}_{hl}_{key}_{stepsize}_{args.type}.json")))

                offset_db = whole_test_set.range_per_height_db[str(hl)]["start"]
                end_db = whole_test_set.range_per_height_db[str(hl)]["end"]

                offset_query = whole_test_set.range_per_height_query[str(hl)]["start"]
                end_query = whole_test_set.range_per_height_query[str(hl)]["end"]
                
                sub_db = whole_test_set.db_image[offset_db:end_db]
                queries_at_height = query_features[offset_query:end_query]

                # precompute faiss indices
                faiss_indices = {}
                # remember mapping from local indices to global ones
                mapping_absolute = {}
                # how large are clusters (at height hl)
                sizes_at_height = []

                print("\t\t\t'->Precomputing faiss indices...")

                for i in tqdm(range(key)):
                    
                    sizes_at_height.append(len(references_cluster[str(i)]))
                    
                    relative_indices = [sub_db.index(f"{hl}/{ref}_{hl}.png")
                        for ref in references_cluster[str(i)]]
                    absolute_indices = list(np.array(relative_indices, dtype=int) + offset_db)

                    faiss_index = faiss.IndexFlatL2(database_features.shape[1])
                    faiss_index.add(np.take(database_features, absolute_indices, axis=0))

                    faiss_indices[i] = faiss_index
                    mapping_absolute[i] = absolute_indices

                top_sizes = np.flip(np.sort(sizes_at_height))

                for strategy in predictions.keys():
                    # keep track of search space sizes
                    search_space_sizes[strategy].append(np.sum(top_sizes[:size_consideration[strategy]]))

                soft_predictions = np.load(join(args.assignment_files_directory,
                    f"sp_{method}_{hl}_{key}_{stepsize}_{args.type}.npy"))
                
                print("\t\t\t'->Evaluating...")

                for qix, sp in tqdm(enumerate(soft_predictions), total=soft_predictions.shape[0]):

                    # for each query: use semantic soft prediction
                    # this is used within different strategies

                    for strategy in predictions.keys():

                        if strategy == "hard-clustering":
                            # only 'best' cluster
                            top = [np.argmax(sp)]
                            
                        elif strategy == "soft-top-2":
                            # best two clusters
                            top = np.argpartition(sp, -2)[-2:]

                        elif strategy == "soft-top-3":
                            # top 3
                            top = np.argpartition(sp, -3)[-3:]

                        elif strategy == "soft-threshold-0.34":
                            # only if gamma >= 0.34 but take
                            # at least one cluster (best)
                            sp_copy = sp.copy()
                            sp_copy[top] = 1.
                            top = np.argmax(sp)
                            top = np.argwhere(sp_copy >= 0.34).flatten()

                        elif strategy == "soft-threshold-0.26":
                            # only if gamma >= 0.26 but take
                            # at least one cluster (best)
                            sp_copy = sp.copy()
                            sp_copy[top] = 1.
                            top = np.argmax(sp)
                            top = np.argwhere(sp_copy >= 0.26).flatten()

                        # for each search space take top 20 
                        # and then mix them together and take top 20 again
                        # i.e. top 20 of best 3 cluster can be found by taking top 20
                        # of the set of al individual top 20s for each cluster
                        all_distances, all_predictions = [], []

                        for cix in top:

                            current_query = np.array([queries_at_height[qix]])

                            distances, prediction = faiss_indices[cix].search(current_query,
                                max(n_values))
                            
                            all_distances.extend(list(distances.flatten()))
                            all_predictions.extend(list(np.take(mapping_absolute[cix], prediction)))

                        top_indices = np.argsort(all_distances)[:max(n_values)]
                        predictions[strategy].append(np.take(all_predictions, top_indices))

            recalls, reductions = {}, {}

            for strategy in predictions.keys():

                correct_at_n = np.zeros(len(n_values))

                # TODO: can we do this on the matrix in one go?
                for qIx, pred in enumerate(predictions[strategy]):

                    pred_flat = pred.flatten()

                    for i, n in enumerate(n_values):

                        # if in top N then also in top NN, where NN > N
                        if np.any(np.in1d(pred_flat[:n], gt[qIx])):
                            correct_at_n[i:] += 1
                            break

                recall_at_n = correct_at_n / whole_test_set.num_queries
                recalls[strategy] = list(recall_at_n)

                reductions[strategy] = 1 - np.max(search_space_sizes[strategy]).item() / whole_test_set.num_per_height_db["300"]

            results[f"{method}_{key}"] = {
                "recalls": recalls,
                "reduction": reductions
            }

    with open(f"results{args.result_tag}.json", "w+") as fout:
        json.dump(results, fout, indent=4)
