import json
import time
import faiss
import argparse
import numpy as np
import fourseasons as dataset

from os.path import *
from tqdm import tqdm

stepsize = 100
n_values = [1, 5, 10, 20]

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
        default="../sem-clustering/results/infered")
    parser.add_argument("--type", type=str,
        help="Type of clustering", choices=["sem", "vpr"],
        default="sem")

    args = parser.parse_args()
    
    print("--- TIMING EVALUATION ---")
    
    query_features = np.load(args.query_features_file)
    database_features = np.load(args.db_features_file)
    
    whole_test_set = dataset.get_whole_val_set()
    gt = whole_test_set.getPositives()
    
    results = dict()
    
    # -------------- vanilla retrieval --------------
    
    print("--> VANILLA RETRIEVAL:")
    
    vanilla_durations = []
    
    for height_level in [300, 400, 500, 600]:
        
        # database_features contains features for each height levels
        offset_db = whole_test_set.range_per_height_db[str(height_level)]["start"]
        end_db = whole_test_set.range_per_height_db[str(height_level)]["end"]
        
        # query_features is also sorted acc. to height levels
        offset_query = whole_test_set.range_per_height_query[str(height_level)]["start"]
        end_query = whole_test_set.range_per_height_query[str(height_level)]["end"]
        
        # get database in variant for current height
        db_at_height = database_features[offset_db:end_db]
        # get only those queries that are at this height
        queries_at_height = query_features[offset_query:end_query]
        
        faiss_index = faiss.IndexFlatL2(database_features.shape[1])
        faiss_index.add(db_at_height)
        
        durations_at_hl = []
        
        for Q in tqdm(queries_at_height, total=queries_at_height.shape[0]):
            
            start = time.time()
            _, _ = faiss_index.search(Q.reshape(1, -1),
                max(n_values))
            end = time.time()
            
            durations_at_hl.append(end-start)
        
        print(f"\n---> avg. duration at hl={height_level}:", np.mean(durations_at_hl))
        results[f"vanilla-{height_level}"] = np.mean(durations_at_hl)
        
        vanilla_durations.extend(durations_at_hl)
            
    print(f"---> avg. duration over all hl:", np.mean(vanilla_durations))
    results["vanilla"] = np.mean(vanilla_durations)
            
    print("--> SEMANTIC CLUSTERING RETRIEVAL:")
    
    for method in my_methods:

        for key in cluster_nums:
            
            print(f"---> evaluating {method} with {key} clusters")
            
            all_durations = []
            
            for hl in height_levels:

                print(f"----> height level {hl}")
                
                durations_at_hl = []

                references_cluster = json.load(open(join(args.assignment_files_directory,
                    f"clusters_{method}_{hl}_{key}_{stepsize}.json")))

                offset_db = whole_test_set.range_per_height_db[str(hl)]["start"]
                end_db = whole_test_set.range_per_height_db[str(hl)]["end"]

                offset_query = whole_test_set.range_per_height_query[str(hl)]["start"]
                end_query = whole_test_set.range_per_height_query[str(hl)]["end"]
                
                sub_db = whole_test_set.db_image[offset_db:end_db]
                queries_at_height = query_features[offset_query:end_query]

                faiss_indices = {}

                for i in tqdm(range(key)):
                    
                    print(f"-----> precomputing faiss index for cluster {i}")
                    
                    relative_indices = [sub_db.index(f"{hl}/{ref}_{hl}.png")
                        for ref in references_cluster[str(i)]]
                    absolute_indices = list(np.array(relative_indices, dtype=int) + offset_db)

                    faiss_index = faiss.IndexFlatL2(database_features.shape[1])
                    faiss_index.add(np.take(database_features, absolute_indices, axis=0))

                    faiss_indices[i] = faiss_index
                    
                soft_predictions = np.load(join(args.assignment_files_directory,
                    f"sp_{method}_{hl}_{key}_{stepsize}.npy"))
                
                print(f"\n-----> retrieve candidates for queries")
                
                for qix, sp in tqdm(enumerate(soft_predictions), total=soft_predictions.shape[0]):
                    
                    cluster_index = np.argmax(sp)
                    
                    Q = np.array([queries_at_height[qix]])
                    
                    start = time.time()
                    _ = faiss_indices[cluster_index].search(Q.reshape(1, -1),
                        max(n_values))
                    end = time.time()
                    
                    durations_at_hl.append(end-start)
                    
                all_durations.extend(durations_at_hl)
                    
                print(f"\n-----> avg. duration at hl={hl}:", np.mean(durations_at_hl))
                results[f"{method}-{key}-{hl}"] = np.mean(durations_at_hl)
                    
            print(f"-----> avg. duration over all hl ({method}, {key}):", np.mean(all_durations))
            results[f"{method}-{key}"] = np.mean(all_durations)

    with open("timing_performance.json", "w+") as fout:
        json.dump(results, fout, indent=4)
