from sklearn.neighbors import NearestNeighbors
from os.path import join, exists
from itertools import chain
from typing import Dict
from PIL import Image

import torchvision.transforms as transforms
import torch.utils.data as data
import torch

import numpy as np
import itertools
import warnings
import json
import h5py

warnings.filterwarnings("ignore")

# minimum distance to be
# considered as negative
# 350 = (250**2 + 250**2)**0.5 - eps,
# and shift of (250, 250) chosen heuristically 
NEG_MIN_DIST = 350

# maximum distance to be
# considered as positive
# 75 = (50**2 + 50**2)**0.5 + eps,
# and shift of (50, 50) chosen heuristically 
POS_MAX_DIST = 75

# NOTE: adapt this
data_dir = "data/fourseasons"

if not exists(data_dir):
    raise FileNotFoundError("data_dir is hardcoded, please adjust!")

def input_transform():

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.47894198025675805],
            std=[0.20632276007133424]),
    ])

def get_whole_training_set(onlyDB=False):
    return WholeDatasetFromFile(which_dataset="train", input_transform=input_transform(),
        onlyDB=onlyDB)

def get_whole_val_set(which_dataset="val"):
    return WholeDatasetFromFile(which_dataset=which_dataset, input_transform=input_transform())

def get_whole_test_set():
    return WholeDatasetFromFile(which_dataset="test", input_transform=input_transform())

def get_training_query_set(margin=0.1):
    return QueryDatasetFromFile(which_dataset="train", input_transform=input_transform(),
        margin=margin)

def get_val_query_set():
    return QueryDatasetFromFile(which_dataset="val", input_transform=input_transform())

class WholeDatasetFromFile(data.Dataset):

    def __init__(self, which_dataset, db_image_file="db_images.json", db_position_file="db_positions.json",
        query_image_file="query_images.json", query_position_file="query_positions.json",
        input_transform=None, onlyDB=False):

        super().__init__()

        self.input_transform = input_transform

        # "train": { 300: [...], 400: [...], ... }, "val": { ... }
        db_dict: Dict[int, list] = json.load(open(join(data_dir, db_image_file)))[which_dataset]
        query_dict: Dict[int, list] = json.load(open(join(data_dir, query_image_file)))[which_dataset]

        assert db_dict.keys() == query_dict.keys(), "Query and db images should not have different height levels"
        self.height_levels = sorted(list(db_dict.keys()))

        db_pos_dict: Dict[int, list] = json.load(open(join(data_dir, db_position_file)))[which_dataset]
        query_pos_dict: Dict[int, list] = json.load(open(join(data_dir, query_position_file)))[which_dataset]

        self.num_per_height_db = {hl: len(db_dict[hl]) for hl in self.height_levels}
        self.num_per_height_query = {hl: len(query_dict[hl]) for hl in self.height_levels}

        # TODO: simplify this
        self.range_per_height_db = {hl: {
            "start": sum([self.num_per_height_db[k] for k in self.height_levels][:self.height_levels.index(hl)]),
            "end": sum([self.num_per_height_db[k] for k in self.height_levels][:self.height_levels.index(hl)+1])
        } for hl in self.height_levels}

        # TODO: simplify this as well
        self.range_per_height_query = {hl: {
            "start": sum([self.num_per_height_query[k] for k in self.height_levels][:self.height_levels.index(hl)]),
            "end": sum([self.num_per_height_query[k] for k in self.height_levels][:self.height_levels.index(hl)+1])
        } for hl in self.height_levels}

        self.db_image = list(chain(*[[join(str(hl), f) for f in db_dict[hl]] for hl in self.height_levels]))
        self.query_image = list(chain(*[[join(str(hl), f) for f in query_dict[hl]] for hl in self.height_levels]))

        self.num_db = len(self.db_image)
        self.num_queries = len(self.query_image)

        self.which_dataset = which_dataset
        self.dataset = "fourseasons"

        dataset_root = join(data_dir, self.which_dataset)

        self.images = [join(dataset_root, "references", img) for img in self.db_image]

        self.num_db = len(self.db_image)
        self.num_queries = len(self.query_image)

        if not onlyDB:
            self.images += [join(dataset_root, "queries", img) for img in self.query_image]

        self.db_position = np.asfarray(list(chain(*[db_pos_dict[hl] for hl in self.height_levels])))
        self.query_position = np.asfarray(list(chain(*[query_pos_dict[hl] for hl in self.height_levels])))

        self.positives = None
        self.distances = None

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert("L")

        # NOTE: sketch for the case that images are only given once in "oversize"
        # i.e., instead of depositing for 300,...,600 one could deposit only 600 
        # but with twice the pixel density s.t. one could crop to 300 and has same pixel res.
        # if index < self.num_db:

        #     # only oversize given
        #     assert len(self.height_levels) == 1

        #     current_level = int(self.height_levels[0])

        #     rel = current_level / 600 # 600 = ref. level (highest)
        #     size = ceil(rel * 2200) # 2200 = "oversize size"
        #     offset = (2200-  size) // 2 

        #     # center crop image to same s.t. square image
        #     image = image.crop((offset, offset, offset+size, offset+size))
        #     image = image.resize((1100, 1100))

        if self.input_transform:
            image = self.input_transform(image)

        return image, index

    def __len__(self):

        return len(self.images)

    def getPositives(self):

        if self.positives is None:

            # TODO: replace with range_per_height_db, range_per_height_query
            index_offset_db, index_offset_query = 0, 0
            
            self.distances = [] # initialize distances
            self.positives = [] # initialize positives

            for hl in self.height_levels:

                index_end_db = index_offset_db + self.num_per_height_db[hl]
                index_end_query = index_offset_query + self.num_per_height_query[hl]

                # positives are those within POS_MAX_DIST range
                # fit NN to find them, search by radius
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(self.db_position[index_offset_db:index_end_db])

                # TODO: maybe include image that are in area between POS_MAX_DIST and NEG_MIN_DIST
                distances_hl, positives_hl = knn.radius_neighbors(self.query_position[index_offset_query:index_end_query],
                    radius=POS_MAX_DIST)

                for index_array in positives_hl:
                    # always return absolute db index
                    self.positives.append(index_array + index_offset_db)

                self.distances.extend(distances_hl)

                index_offset_db = index_end_db
                index_offset_query = index_end_query
        
        return self.positives

def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).
    
    Args:
        data: list of tuple (query, positive, negatives). 
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter (lambda x: x is not None, batch))

    if len(batch) == 0:
        return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    negatives = torch.cat(negatives, 0)
    
    indices = list(itertools.chain(*indices))

    return query, positive, negatives, negCounts, indices

class QueryDatasetFromFile(data.Dataset):

    def __init__(self, which_dataset, db_image_file="db_images.json", db_position_file="db_positions.json",
        query_image_file="query_images.json", query_position_file="query_positions.json",
        num_neg_samples=1000, num_neg=10, margin=0.1, input_transform=None):

        super().__init__()

        self.input_transform = input_transform
        self.margin = margin

        self.which_dataset = which_dataset
        self.dataset = "fourseasons"
        
        # number of negatives to randomly sample
        self.num_neg_samples = num_neg_samples
        # number of negatives used for training
        self.num_neg = num_neg

        # "train": { 300: [...], 400: [...], ... }, "val": { ... }
        db_dict: Dict[int, list] = json.load(open(join(data_dir, db_image_file)))[which_dataset]
        query_dict: Dict[int, list] = json.load(open(join(data_dir, query_image_file)))[which_dataset]

        assert db_dict.keys() == query_dict.keys(), "Query and db images should not have different height levels"
        height_levels = sorted(list(db_dict.keys()))

        db_pos_dict: Dict[int, list] = json.load(open(join(data_dir, db_position_file)))[which_dataset]
        query_pos_dict: Dict[int, list] = json.load(open(join(data_dir, query_position_file)))[which_dataset]

        self.num_per_height_db = {hl: len(db_dict[hl]) for hl in height_levels}
        self.num_per_height_query = {hl: len(query_dict[hl]) for hl in height_levels}

        self.db_image = list(chain(*[[join(str(hl), f) for f in db_dict[hl]] for hl in height_levels]))
        self.query_image = list(chain(*[[join(str(hl), f) for f in query_dict[hl]] for hl in height_levels]))

        self.num_db = len(self.db_image)
        self.num_queries = len(self.query_image)

        self.db_position = np.asfarray(list(chain(*[db_pos_dict[hl] for hl in height_levels])))
        self.query_position = np.asfarray(list(chain(*[query_pos_dict[hl] for hl in height_levels])))

        # tricky adaption due to assertion, normally queries w/o positive
        # references would have been sorted out, but now there aren't any
        self.queries = np.arange(self.num_queries)

        index_offset_db, index_offset_query = 0, 0
        
        self.positives = [] # initialize positives
        self.negatives = [] # initialize negatives

        for hl in height_levels:

            index_end_db = index_offset_db + self.num_per_height_db[hl]
            index_end_query = index_offset_query + self.num_per_height_query[hl]

            # positives are those within POS_MAX_DIST range
            # fit NN to find them, search by radius
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.db_position[index_offset_db:index_end_db])

            positives_hl = knn.radius_neighbors(self.query_position[index_offset_query:index_end_query],
                radius=POS_MAX_DIST, return_distance=False)

            # radius returns unsorted, sort once now so we dont have to later
            for i, pos in enumerate(positives_hl):

                # assert that each query has at least one positive
                assert len(pos) > 0, f"Query with index {i + index_offset_query} does not have a positive reference"

                self.positives.append((np.sort(pos) + index_offset_db).tolist())

            inside_negative_radius = knn.radius_neighbors(self.query_position[index_offset_query:index_end_query],
                radius=NEG_MIN_DIST, return_distance=False)
            
            for i, complementary_negative in enumerate(inside_negative_radius):

                negatives = np.setdiff1d(np.arange(self.num_db), complementary_negative + index_offset_db, assume_unique=True)

                assert len(negatives) >= num_neg_samples, f"Query with index {i + index_offset_query} only has {len(negatives)} negatives, \
                    {num_neg_samples} are specified to be sampled"
                
                # for a query, negatives are all images in db except that within NEG_MIN_DIST
                self.negatives.append(negatives)

            index_offset_db = index_end_db
            index_offset_query = index_end_query

        # filepath of HDF5 containing feature vectors for images
        self.cache = None
        self.negative_cache = [np.empty((0,)) for _ in range(self.num_queries)]

    def __getitem__(self, index):

        # re-map index to match dataset
        index = self.queries[index]

        with h5py.File(self.cache, mode="r") as h5:

            h5feat = h5.get("features")

            # queries are stored 'behind' db
            query_offset = len(self.db_position)
            query_features = h5feat[index + query_offset]

            # get features of all positives for current query
            positives_features = h5feat[self.positives[index]]

            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(positives_features)

            distance_positives, indices_positives_features = knn.kneighbors(query_features.reshape(1,-1), 1)
            distance_positives = distance_positives.item()

            # get positive element for query that has greatest 'feature similarity'
            index_selected_positive = self.positives[index][indices_positives_features[0].item()]

            # first samples num_neg_samples many negatives for query
            negative_samples = np.random.choice(self.negatives[index], self.num_neg_samples, replace=False)
            negative_samples = np.unique(np.concatenate([self.negative_cache[index], negative_samples]))

            negative_features = h5feat[negative_samples.astype(int).tolist()]
            knn.fit(negative_features)

            distance_negatives, indices_negative_features = knn.kneighbors(query_features.reshape(1,-1), 
                self.num_neg * 10) # to quote netvlad paper code: 10x is hacky but fine

            distance_negatives = distance_negatives.reshape(-1)
            indices_negative_features = indices_negative_features.reshape(-1)

            # try to find negatives that are within margin, if there aren't any return none
            violating_negatives = distance_negatives < distance_positives + self.margin**0.5
     
            if np.sum(violating_negatives) < 1:
                # if none are violating then skip this query
                return None

            indices_negative_features = indices_negative_features[violating_negatives][:self.num_neg]
            negative_indices = negative_samples[indices_negative_features].astype(np.int32)
            self.negative_cache[index] = negative_indices

        dataset_root = join(data_dir, self.which_dataset)

        query = Image.open(join(dataset_root, "queries", self.query_image[index])).convert("L")
        positive = Image.open(join(dataset_root, "references", self.db_image[index_selected_positive])).convert("L")

        if self.input_transform:
            query = self.input_transform(query)
            positive = self.input_transform(positive)

        negatives = []

        for neg_idx in negative_indices:

            negative = Image.open(join(dataset_root, "references", self.db_image[neg_idx])).convert("L")

            if self.input_transform:
                negative = self.input_transform(negative)

            negatives.append(negative)

        negatives = torch.stack(negatives, 0)

        return query, positive, negatives, [index, index_selected_positive] + negative_indices.tolist()

    def __len__(self):

        return len(self.queries)
