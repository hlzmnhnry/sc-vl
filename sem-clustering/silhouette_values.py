from cluster import methods
from sklearn.metrics import silhouette_score
from os.path import *
from os import mkdir, sep

import argparse
import json

# tag for overview
stepsize = 100

# hard-coded for experiments
height_levels = [300, 400, 500, 600]
my_methods = ["SameSizeKMeans", "KMeans", "GaussianMixture"]
cluster_nums = range(3, 10+1)

output_dir = "silhouette"

def path_to_string_without_slashes(path):
    
    # normalize the path
    normalized_path = normpath(path)
    
    # remove the path separator
    cleaned_path = normalized_path.replace(sep, "_")
    
    return cleaned_path

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute silhouette values for learned clusters", add_help=True)

    parser.add_argument("--type", type=str, required=False,
        help="Type of descriptors within learned clusters", choices=["sem", "vpr"],
        default="sem")
    parser.add_argument("--pickle_path", type=str, required=False,
        help="Path to learned cluster models", default="pickle/normal")
    
    args = parser.parse_args()
    
    silhouette_scores = dict()
    size_at_hl = dict()
    
    for method in my_methods:
        
        print("method:", method)

        cm: methods.ClusterMethod = getattr(methods, method)
        
        silhouette_scores[method] = dict()
        size_at_hl[method] = dict()

        for key in cluster_nums:
            
            print("k:", key)
            
            silhouette_scores[method][key] = dict()
            size_at_hl[method][key] = dict()
            offset = 0
            
            for hl in height_levels:
                
                print("hl:", hl)

                loading_key = f"{hl}_{key}_{stepsize}_{args.type}"
                instance: methods.ClusterMethod = cm.load_model(key=loading_key,
                    model_directory=args.pickle_path)
                
                score = silhouette_score(instance.data, instance.labels, metric="euclidean")
                
                silhouette_scores[method][key][hl] = score
                size_at_hl[method][key][hl] = instance.labels.shape[0]
    
    if not exists(output_dir): mkdir(output_dir)
    
    fname = join(output_dir, f"{path_to_string_without_slashes(args.pickle_path)}.json")
    
    with open(fname, "w+") as fout:
        json.dump({
            "silhouette_scores": silhouette_scores,
            "size_at_hl": size_at_hl
            }, fout, indent=4)
        
    print(f"Wrote silhouette scores to {fname}")
