import time
import json
import argparse
import numpy as np

from os import mkdir
from tqdm import tqdm
from os.path import *
from cluster import methods
from datetime import timedelta
from typing import List, Dict, Tuple

# tag for overview
stepsize = 100

# hard-coded for experiments
height_levels = [300, 400, 500, 600]
my_methods = ["SameSizeKMeans", "KMeans", "GaussianMixture"]
cluster_nums = range(3, 6+1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute reference clusters", add_help=True)

    parser.add_argument("--input_file", type=str, required=True,
        help="Path to input file containing (semantic/vpr) descriptors")
    parser.add_argument("--type", type=str, required=True,
        help="Type of descriptors in input file", choices=["sem", "vpr"])
    parser.add_argument("--db_positions", type=str, 
        help="File containing positions of db images",
        default="../pytorch-NetVlad/data/fourseasons/db_positions.json")

    args = parser.parse_args()

    if args.type == "sem":
        # json file containing semantic descriptors 
        # follows form: 
        # {
        # "300": {
        #     "375900_5600800": [
        #         1.0,  
        #         0.0,
        #         0.0,
        #         0.0
        #     ], ...
        # 
        input_data = json.load(open(args.input_file))
    elif args.type == "vpr":
        # these are NetVLAD descriptors in shape (N, D)
        # but N is divided into H height levels
        input_data = np.load(args.input_file)

    # load positions of validation DB
    positions = json.load(open(args.db_positions))["val"]
    offset = 0

    # make output directories 
    if not exists("pickle"): mkdir("pickle")
    if not exists("centers"): mkdir("centers")
    
    # height levels are fix
    for hl in height_levels:

        assert type(positions[str(hl)]) == list

        if args.type == "sem":
            # descriptors at current hl
            data_at_hl = input_data[hl]
            # bring them in order
            data = np.asfarray([data_at_hl[f"{x}_{y}"] for x, y in positions[str(hl)]])
        elif args.type == "vpr":
            # assumes that vpr data is ordered regarding height levels
            l = len(positions[str(hl)])
            data = input_data[offset:offset+l]

        annotations = [f"{x}_{y}" for (x, y) in positions[str(hl)]]

        for method in my_methods:

            cm: methods.ClusterMethod = getattr(methods, method)

            for key in cluster_nums:

                key_str = "" if key is None else f"with n_cluster={key}"
                print(f"\nPrecomputing {cm.name} {key_str} at height={hl}...")

                save_key = f"{hl}_{key}_{stepsize}_{args.type}"

                cluster_args = {
                    "data": data,
                    "annotations": annotations
                }

                if key is not None: cluster_args["n_clusters"] = key

                start_time = time.time()
                instance: methods.ClusterMethod = cm(**cluster_args)
                end_time = time.time()

                print(f"Took {timedelta(seconds=end_time-start_time)}")
                instance.save(key=save_key)

        if args.type == "vpr": offset += l
