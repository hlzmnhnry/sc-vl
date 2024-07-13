import json
import time
import argparse
import numpy as np

from datetime import timedelta
from cluster import methods
from os.path import *

# tag for overview
stepsize = 100

# hard-coded for experiments
height_levels = [300, 400, 500, 600]
my_methods = ["SameSizeKMeans", "KMeans", "GaussianMixture"]
cluster_nums = range(3, 6+1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="(Soft) assign queries to clusters", add_help=True)

    parser.add_argument("--input_file", type=str, required=True,
        help="Path to input file containing (semantic) descriptors of queries")
    parser.add_argument("--type", type=str, required=True,
        help="Type of descriptors in input file", choices=["sem", "vpr"])
    parser.add_argument("--db_positions", type=str, 
        help="File containing positions of db images",
        default="../pytorch-NetVlad/data/fourseasons/db_positions.json")
    parser.add_argument("--order_file", type=str,
        help="Path to file listing order of queries (important not to mix order)",
        default="../pytorch-netvlad/data/fourseasons/v1/query_images.json")

    args = parser.parse_args()

    if args.type == "sem":
        # json file containing semantic descriptors 
        # follows form: 
        # {
        # "2023-06-12-12-24-32_357397": [
        #     0.0,  
        #     0.0,
        #     0.0,
        #     1.0
        # ], ...
        # 
        input_data = json.load(open(args.input_file))
    elif args.type == "vpr":
        # these are NetVLAD descriptors in shape (N, D)
        # but N is divided into H height levels
        input_data = np.load(args.input_file)

    order = json.load(open(args.order_file))["val"]
    order = {hl: [splitext(f)[0] for f in order[str(hl)]] for hl in height_levels}

    for method in my_methods:

        cm: methods.ClusterMethod = getattr(methods, method)

        for key in cluster_nums:

            offset = 0
            
            for hl in height_levels:

                if args.type == "sem":
                    # descriptors at current hl
                    data_at_hl = [input_data[o] for o in order[hl]]
                    # bring them in order
                    data = np.asfarray(data_at_hl)
                elif args.type == "vpr":
                    # assumes that vpr data is ordered regarding height levels
                    l = len(order[hl])
                    data = input_data[offset:offset+l]

                loading_key = f"{hl}_{key}_{stepsize}_{args.type}"
                instance: methods.ClusterMethod = cm.load_model(key=loading_key,
                    model_directory="pickle")
                
                print(f"\nAssigning queries at height={hl} to one of {key} clusters learned with {cm.name}...")

                start_time = time.time()
                soft_query_prediction = instance.soft_predict(data)
                end_time = time.time()

                print(f"Took {timedelta(seconds=end_time-start_time)}")
                np.save(f"sp_{method}_{loading_key}.npy", soft_query_prediction)

                # manage list of elements for each cluster
                cluster_elements = {idx: [] for idx in range(instance.n_clusters)}

                for qix in range(len(instance.labels)):
                    cluster_elements[instance.labels[qix]].append(instance.annotations[qix])
                
                with open(f"clusters_{method}_{loading_key}.json", "w+") as fout:
                    json.dump(cluster_elements, fout, indent=4)

                if args.type == "vpr": offset += l
