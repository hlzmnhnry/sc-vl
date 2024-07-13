import matplotlib.pyplot as plt
import sklearn.cluster as sc
import sklearn.mixture
import sklearn.metrics
import numpy as np
import warnings
import pickle
import abc

from sklearn.metrics.pairwise import euclidean_distances

from sklearn.neighbors import NearestNeighbors
from typing import Union, List
from random import randint
from os import makedirs
from os.path import *
from tqdm import tqdm

class ClusterMethod(metaclass=abc.ABCMeta):

    # class variable
    # flag whether method has a parameter
    # for the number of clusters
    parameterized_cluster_count: bool
    # class variable
    # name of concrete method
    name: str

    # data used for clustering
    # in the same order as labels/annotations
    data: np.ndarray
    # assigned labels of train points
    # in the same order as data/annotations
    labels: np.ndarray
    # annotations to data
    # in the same order as labels/data
    annotations: Union[List[str], None]
    # number of clusters
    n_clusters: int

    def __init__(self, data: np.ndarray, n_clusters: int, labels: np.ndarray,
        annotations: Union[List[str], None] = None) -> None:

        assert data.shape[0] == labels.shape[0], "Each data point should have exactly one label"
        assert np.sum(np.unique(labels) + 1) == (n_clusters**2 + n_clusters) / 2, "There is at least one empty cluster"

        self.data = data
        self.n_clusters = n_clusters
        self.labels = labels
        self.annotations = annotations

    @abc.abstractmethod
    def soft_predict(self, x: np.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: np.ndarray):
        raise NotImplementedError
    
    def save(self, key: Union[int, str], model_directory="pickle", center_directory="centers"):

        if not exists: makedirs(model_directory)
        out_name = join(model_directory, f"{self.name}_model_{key}.pkl")

        with open(out_name, "wb") as fout:
            pickle.dump(self, fout)

        if hasattr(self, "centers"):
            if not exists: makedirs(center_directory)
            out_name = join(center_directory, f"{self.name}_centers_{key}.txt")
            np.savetxt(out_name, self.centers)

    @classmethod
    def load_model(cls, key: Union[int, str], model_directory="pickle"):
        return pickle.load(open(join(model_directory, f"{cls.name}_model_{key}.pkl"), "rb"))

class KMeans(ClusterMethod):

    parameterized_cluster_count = True
    name = "KMeans"

    def __init__(self, data: np.ndarray, n_clusters: int, annotations: Union[List[str], None] = None,
        **kwargs) -> None:

        self.km = sc.KMeans(n_clusters=n_clusters, **kwargs).fit(data)
        self.centers = self.km.cluster_centers_

        super().__init__(data=data, n_clusters=n_clusters, labels=self.km.labels_, annotations=annotations)

    def soft_predict(self, x: np.ndarray):

        n, _ = x.shape

        data = np.repeat(x[:, :, np.newaxis], self.n_clusters, axis=2)
        dist = np.linalg.norm(data - np.repeat(self.centers.T[np.newaxis, :, :], n, axis=0), axis=1)

        rows_sum = dist.sum(axis=1)
        gamma = 1 - (dist / rows_sum[:, np.newaxis])

        # normalize again
        gamma = gamma / gamma.sum(axis=1)[:, np.newaxis]

        return gamma

    def predict(self, x: np.ndarray):

        prediction = self.km.predict(x).flatten()
        
        if len(prediction) == 1:
            return prediction.item()

        return prediction

class BisectingKMeans(ClusterMethod):

    parameterized_cluster_count = True
    name = "BisectingKMeans"

    def __init__(self, data: np.ndarray, n_clusters: int, annotations: Union[List[str], None] = None,
        **kwargs) -> None:

        self.bkm = sc.BisectingKMeans(n_clusters=n_clusters, **kwargs).fit(data)
        self.centers = self.bkm.cluster_centers_

        super().__init__(data=data, n_clusters=n_clusters, labels=self.bkm.labels_, annotations=annotations)

    def soft_predict(self, x: np.ndarray):

        n, _ = x.shape

        data = np.repeat(x[:, :, np.newaxis], self.n_clusters, axis=2)
        dist = np.linalg.norm(data - np.repeat(self.centers.T[np.newaxis, :, :], n, axis=0), axis=1)

        rows_sum = dist.sum(axis=1)
        gamma = 1 - (dist / rows_sum[:, np.newaxis])

        # normalize again
        gamma = gamma / gamma.sum(axis=1)[:, np.newaxis]

        return gamma

    def predict(self, x: np.ndarray):

        prediction = self.bkm.predict(x).flatten()
        
        if len(prediction) == 1:
            return prediction.item()

        return prediction

class Birch(ClusterMethod):

    parameterized_cluster_count = False
    name = "Birch"

    def __init__(self, data: np.ndarray, n_clusters=None, annotations: Union[List[str], None] = None,
        **kwargs) -> None:

        self.brc = sc.Birch(n_clusters=n_clusters, **kwargs).fit(data)
        self.centers = self.brc.subcluster_centers_
        
        n_clusters = len(self.centers)
        super().__init__(data=data, n_clusters=n_clusters, labels=self.brc.labels_, annotations=annotations)
    
    def predict(self, x: np.ndarray):

        prediction = self.brc.predict(x).flatten()
        
        if len(prediction) == 1:
            return prediction.item()

        return prediction

class Optics(ClusterMethod):

    parameterized_cluster_count = False
    name = "Optics"

    def __init__(self, data: np.ndarray, prediction_nn=1000, annotations: Union[List[str], None] = None,
        **kwargs) -> None:

        self.opt = sc.OPTICS(min_samples=0.05, n_jobs=-1, **kwargs).fit(data)
        labels = self.opt.labels_

        # +1 for indexing start at 0, -1 for unassigned
        n_clusters = len(np.unique(labels)) - 1
        
        # get all assigned points, learn 1-NN classifier
        indices_assigned = np.argwhere(labels != -1).flatten()
        data_assigned = np.take(data, indices_assigned, axis=0)
        nn_assigned = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(data_assigned)

        # get all unassigned points and assign them a label
        # according to 1-NN classifier learnt above
        indices_unassigned = np.argwhere(labels == -1).flatten()
        data_unassigned = np.take(data, indices_unassigned, axis=0)
        assignments_unassigned = nn_assigned.kneighbors(data_unassigned, return_distance=False).flatten()
        true_indices = np.take(indices_assigned, assignments_unassigned)
        labels[indices_unassigned] = np.take(labels, true_indices)

        self.nn_prediction = NearestNeighbors(n_neighbors=prediction_nn, n_jobs=-1).fit(data)

        super().__init__(data=data, n_clusters=n_clusters, labels=labels, annotations=annotations)

    def soft_predict(self, x: np.ndarray):

        self.nn_prediction = NearestNeighbors(n_neighbors=1000, n_jobs=-1).fit(self.data)
        
        # find nearest neighbors of x in ref. points
        indices = self.nn_prediction.kneighbors(x, return_distance=False)
        
        # clusters in neighborhood
        nearest_clusters = np.take(self.labels, indices)

        # the following block finds most frequent cluster index
        # for each row in nearest_clusters and returns them
        N, M = nearest_clusters.shape[0], np.max(nearest_clusters)+1
        bincount_2D = np.zeros(shape=(N, M), dtype=int)
        advanced_indexing = np.repeat(np.arange(N), nearest_clusters.shape[1]), nearest_clusters.ravel()
        np.add.at(bincount_2D, advanced_indexing, 1)

        bincount_2D = np.pad(bincount_2D, [(0, 0), (0, self.n_clusters - bincount_2D.shape[1])],
            mode="constant", constant_values=0)
        
        rows_sum = bincount_2D.sum(axis=1)
        gamma = bincount_2D / rows_sum[:, np.newaxis]
        
        return gamma

    def predict(self, x: np.ndarray):
        
        soft = self.soft_predict(x)
        prediction = np.argmax(soft, axis=1)
        
        if len(prediction) == 1:
            return prediction.item()

        return prediction

class MeanShift(ClusterMethod):

    parameterized_cluster_count = False
    name = "MeanShift"

    def __init__(self, data: np.ndarray, annotations: Union[List[str], None] = None, **kwargs) -> None:

        self.ms = sc.MeanShift(**kwargs).fit(data)
        self.centers = self.ms.cluster_centers_
        
        n_clusters = self.centers.shape[0]
        super().__init__(data=data, n_clusters=n_clusters, labels=self.ms.labels_, annotations=annotations)

    def soft_predict(self, x: np.ndarray):

        n, _ = x.shape

        data = np.repeat(x[:, :, np.newaxis], self.n_clusters, axis=2)
        dist = np.linalg.norm(data - np.repeat(self.centers.T[np.newaxis, :, :], n, axis=0), axis=1)

        rows_sum = dist.sum(axis=1)
        gamma = 1 - (dist / rows_sum[:, np.newaxis])

        # normalize again
        gamma = gamma / gamma.sum(axis=1)[:, np.newaxis]

        return gamma

    def predict(self, x: np.ndarray):
        
        prediction = self.ms.predict(x).flatten()

        if len(prediction) == 1:
            return prediction.item()
        
        return prediction

class GaussianMixture(ClusterMethod):

    parameterized_cluster_count = True
    name = "GaussianMixture"

    def __init__(self, data: np.ndarray, n_clusters: int, annotations: Union[List[str], None] = None,
        **kwargs) -> None:

        self.gm = sklearn.mixture.GaussianMixture(n_components=n_clusters, **kwargs).fit(data)
        self.centers = self.gm.means_
        
        labels = self.gm.predict(data).flatten()
        n_clusters = self.centers.shape[0]
        super().__init__(data=data, n_clusters=n_clusters, labels=labels, annotations=annotations)

    def soft_predict(self, x: np.ndarray):

        proba = self.gm.predict_proba(x)

        rows_sum = proba.sum(axis=1)
        gamma = proba / rows_sum[:, np.newaxis]

        return gamma

    def predict(self, x: np.ndarray):

        a = self.gm.predict_proba(x)
        
        prediction = self.gm.predict(x).flatten()

        if len(prediction) == 1:
            return prediction.item()
        
        return prediction

class SpectralClustering(ClusterMethod):

    parameterized_cluster_count = True
    name = "SpectralClustering"

    def __init__(self, data: np.ndarray, n_clusters: int, prediction_nn=1000,
        annotations: Union[List[str], None] = None, **args) -> None:

        self.sc = sc.SpectralClustering(n_clusters=n_clusters, **args).fit(data)

        super().__init__(data=data, n_clusters=n_clusters, labels=self.sc.labels_, annotations=annotations)

        self.nn_prediction = NearestNeighbors(n_neighbors=prediction_nn, n_jobs=-1).fit(data)

    def soft_predict(self, x: np.ndarray):

        self.nn_prediction = NearestNeighbors(n_neighbors=1000, n_jobs=-1).fit(self.data)
        
        # find nearest neighbors of x in ref. points
        indices = self.nn_prediction.kneighbors(x, return_distance=False)
        
        # clusters in neighborhood
        nearest_clusters = np.take(self.labels, indices)

        # the following block finds most frequent cluster index
        # for each row in nearest_clusters and returns them
        N, M = nearest_clusters.shape[0], np.max(nearest_clusters)+1
        bincount_2D = np.zeros(shape=(N, M), dtype=int)
        advanced_indexing = np.repeat(np.arange(N), nearest_clusters.shape[1]), nearest_clusters.ravel()
        np.add.at(bincount_2D, advanced_indexing, 1)

        bincount_2D = np.pad(bincount_2D, [(0, 0), (0, self.n_clusters - bincount_2D.shape[1])],
            mode="constant", constant_values=0)
        
        rows_sum = bincount_2D.sum(axis=1)
        gamma = bincount_2D / rows_sum[:, np.newaxis]
        
        return gamma

    def predict(self, x: np.ndarray):
        
        soft = self.soft_predict(x)
        prediction = np.argmax(soft, axis=1)
        
        if len(prediction) == 1:
            return prediction.item()

        return prediction

class IterativeNN(ClusterMethod):

    parameterized_cluster_count = True
    name = "IterativeNN"

    @staticmethod
    def precompute_mean_distances(points: np.ndarray, pieces: int):

        if pieces > points.shape[0]: pieces = points.shape[0]

        mean_distances = np.zeros(points.shape[0])
        piece_size = points.shape[0] // pieces

        for p in range(pieces):

            start = p * piece_size
            end = (p + 1) * piece_size if (p + 1) * piece_size <= points.shape[0] else points.shape[0]

            # possibility to compute pairwise distances in pieces 
            # since for large datasets this becomes impossible
            distances = euclidean_distances(points[start:end], points)    
            mean_distances[start:end] = np.mean(distances, axis=1)

        return mean_distances

    def get_next_point(self, points: np.ndarray, indices: np.ndarray, policy: str):
        
        if policy == "random":
            return points[randint(0, points.shape[0] - 1)]
        
        if policy == "min_mean_dist":
            argmin = np.argmin(self.mean_distances[indices])
            return points[argmin]

        if policy == "max_mean_dist":
            argmax = np.argmax(self.mean_distances[indices])
            return points[argmax]
        
        raise ValueError(f"Unkown policy to select next point: {policy}")

    def determine_clusters(self, data: np.ndarray, n_clusters: int, selection_policy: str):

        cluster_index = 0
        points = data.copy()
        true_indices = np.arange(0, points.shape[0], 1, dtype=int)
        assignment = np.zeros(points.shape[0], dtype=int)

        while points.shape[0] >= self.size:
            
            # select point for next cluster center
            next_point = self.get_next_point(points, true_indices, selection_policy)

            # find 'size' many nearest neighbors to next_point
            # and put them together in a new cluster
            nn = NearestNeighbors(n_jobs=-1, n_neighbors=self.size).fit(points)
            _, indices = nn.kneighbors([next_point])

            assignment[true_indices[indices]] = cluster_index
            true_indices = np.delete(true_indices, indices, axis=0)
            points = np.delete(points, indices, axis=0)
            
            cluster_index += 1

        assert max(assignment) + 1 == n_clusters

        return assignment

    def __init__(self, data: np.ndarray, n_clusters: int, prediction_nn=1000,
        selection_policy="max_mean_dist", mean_distances=None, precomputation_pieces=1000,
        annotations: Union[List[str], None] = None) -> None:

        assert prediction_nn <= data.shape[0]

        self.size = data.shape[0] // n_clusters

        if mean_distances is None:
            mean_distances = IterativeNN.precompute_mean_distances(points=data,
                pieces=precomputation_pieces)

        self.prediction_nn = prediction_nn
        self.mean_distances = mean_distances
        labels = self.determine_clusters(data=data, n_clusters=n_clusters,
            selection_policy=selection_policy)

        super().__init__(data=data, n_clusters=n_clusters, labels=labels, annotations=annotations)

        self.nn_prediction = NearestNeighbors(n_neighbors=prediction_nn, n_jobs=-1).fit(data)

    def soft_predict(self, x: np.ndarray):
        
        self.nn_prediction = NearestNeighbors(n_neighbors=1000, n_jobs=-1).fit(self.data)

        # find nearest neighbors of x in ref. points
        indices = self.nn_prediction.kneighbors(x, return_distance=False)
        
        # clusters in neighborhood
        nearest_clusters = np.take(self.labels, indices)

        # the following block finds most frequent cluster index
        # for each row in nearest_clusters and returns them
        N, M = nearest_clusters.shape[0], np.max(nearest_clusters)+1
        bincount_2D = np.zeros(shape=(N, M), dtype=int)
        advanced_indexing = np.repeat(np.arange(N), nearest_clusters.shape[1]), nearest_clusters.ravel()
        np.add.at(bincount_2D, advanced_indexing, 1)

        bincount_2D = np.pad(bincount_2D, [(0, 0), (0, self.n_clusters - bincount_2D.shape[1])],
            mode="constant", constant_values=0)
        
        rows_sum = bincount_2D.sum(axis=1)
        gamma = bincount_2D / rows_sum[:, np.newaxis]
        
        return gamma

    def predict(self, x: np.ndarray):
        
        soft = self.soft_predict(x)
        prediction = np.argmax(soft, axis=1)
        
        if len(prediction) == 1:
            return prediction.item()

        return prediction

class SameSizeKMeans(ClusterMethod):

    parameterized_cluster_count = True
    name = "SameSizeKMeans"

    def initialize(self, data: np.ndarray, n_clusters: int):

        # keep track of indices of data points for each cluster
        # NOTE: thus, it is important that data is not changed
        self.clusters = [[] for _ in range(n_clusters)]
        # vice versa also track for each point to which cluster 
        # it its currently assigned, default '-1' means 'not assigned'
        assignment = -1 * np.ones(data.shape[0], dtype=int)

        # initialize means by kmeans++
        self.centers, _ = sc.kmeans_plusplus(data, n_clusters)
        # copy data/means since rows are deleted
        centers, points = self.centers.copy(), data.copy()
        # keep a mapping of mean/points indices, since clusters are ignored if they are full
        # and points should obviously not assigned twice
        center_indices = np.arange(0, n_clusters, 1, dtype=int)
        point_indices = np.arange(0, data.shape[0], 1, dtype=int)

        while points.shape[0] > data.shape[0] % self.size:

            # tansform data from N x C array to K x N x C by repeating along new axis
            # then for each point calculate distance to each center
            dist = np.repeat(points[np.newaxis, :, :], centers.shape[0], axis=0) - \
                centers.reshape((centers.shape[0], 1, points.shape[1]))
            dist = np.abs(np.linalg.norm(dist, axis=2))

            # preferred clusters for each point
            preferred = np.argmin(dist, axis=0)
            # benefit of best over worst assignment
            benefit = np.amin(dist, axis=0) - np.amax(dist, axis=0)
            order = benefit.argsort()
            
            # save which points are assigned
            new_assigned = []
            
            for index in order:
                # get true index of points
                point_index = point_indices[index]
                # get true index of preferred cluster
                preferred_cluster = center_indices[preferred[index]]

                if len(self.clusters[preferred_cluster]) >= self.size:
                    # if a cluster is full, take it as it is 
                    # and remove the corresponding points and its mean
                    centers = np.delete(centers, preferred[index], axis=0)
                    center_indices = np.delete(center_indices, preferred[index], axis=0)
                    break

                self.clusters[preferred_cluster].append(point_index)
                assignment[point_index] = preferred_cluster
                new_assigned.append(index)

            # remove already assigned points and their indices
            points = np.delete(points, new_assigned, axis=0)
            point_indices = np.delete(point_indices, new_assigned, axis=0)

        # check if each cluster has exactly size many elements
        assert (assignment + 1).sum() == self.size * ((n_clusters**2 + n_clusters) // 2)

        return assignment, point_indices.flatten()

    def swap_clusters(self, assignment: np.ndarray, point1_idx: int, point1_cluster: int,
        point2_idx: int, point2_cluster: int):

        self.clusters[point1_cluster].remove(point1_idx)
        self.clusters[point2_cluster].remove(point2_idx)

        self.clusters[point1_cluster].append(point2_idx)
        self.clusters[point2_cluster].append(point1_idx)

        assignment[point1_idx] = point2_cluster
        assignment[point2_idx] = point1_cluster

        return assignment

    def determine_clusters(self, data: np.ndarray, n_clusters: int, inital_assignment: np.ndarray,
        max_iterations=1000):

        assignment = inital_assignment.copy()

        for _ in range(max_iterations):

            # recompute cluster means
            self.centers = np.mean(np.take(data, self.clusters, axis=0), axis=1)
            # keep list of point indices which would rather be in an other cluster
            desire_to_leave = [[] for _ in range(n_clusters)]
            
            # for each point determine distance to new means
            dist = np.repeat(data[np.newaxis, :, :], self.centers.shape[0], axis=0) - \
                self.centers.reshape((self.centers.shape[0], 1, data.shape[1]))
            dist = np.abs(np.linalg.norm(dist, axis=2))

            # delta is difference of distance to best possible cluster center 
            # and distance to currently assigned cluster center
            delta = np.amin(dist, axis=0) - np.choose(np.where(assignment < 0, 0, assignment), dist)
            order = delta.argsort()
            
            # track for each point whether it
            # already has been moved/swapped
            already_moved = [False] * len(order)

            for index in tqdm(order):

                if already_moved[index] or assignment[index] < 0:
                    # do not swap an element twice in one iteration
                    # and do not consider unassigned elements
                    continue
                
                # iterate over all other cluster indices, sorted by their distance
                for other_cluster in np.setdiff1d(dist[:, index].argsort(), [assignment[index]]):

                    # check if there is an element which wants
                    # to leave the other cluster 
                    if len(desire_to_leave[other_cluster]) > 0:

                        # current value defined by distance of index to its cluster plus distance of each candidate to its cluster
                        current = np.take(dist[other_cluster, :], desire_to_leave[other_cluster]) + dist[assignment[index], index]
                        # swap value defined by distance of index to new cluster plus distance of each candidate to its new cluster
                        swap = np.take(dist[assignment[index], :], desire_to_leave[other_cluster]) + dist[other_cluster, index]

                        # get where improvement
                        mask = (swap < current)

                        if np.any(mask):

                            swap_idx = np.argmin(swap - current)
                            candidate = desire_to_leave[other_cluster][swap_idx]

                            # swap clusters
                            assignment = self.swap_clusters(assignment, index,
                                assignment[index], candidate, other_cluster)

                            # remove both elements from outgoing transfer list
                            if index in desire_to_leave[assignment[index]]:
                                desire_to_leave[assignment[index]].remove(index)
                            desire_to_leave[other_cluster].remove(candidate)

                            # flag that both elements have already been moved
                            already_moved[candidate], already_moved[index] = True, True
                            break

                # element has not been moved
                if not already_moved[index]:
                    if index not in desire_to_leave[assignment[index]]:
                        # add it to transfer list
                        desire_to_leave[assignment[index]].append(index)

            # check if each cluster has exactly size many elements
            assert (assignment + 1).sum() == self.size * ((n_clusters**2 + n_clusters) // 2)

            print("already_moved", sum(already_moved))

            # no more transfers were done
            if sum(already_moved) == 0: return assignment
        
        # update cluster means
        self.centers = np.mean(np.take(data, self.clusters, axis=0), axis=1)

        # max iterations reached
        return assignment

    def __init__(self, data: np.ndarray, n_clusters: int, prediction_nn=1000,
        annotations: Union[List[str], None] = None) -> None:

        self.size = data.shape[0] // n_clusters

        # assert prediction_nn <= data.shape[0]

        if data.shape[0] % self.size != 0:
            warnings.warn(f"{data.shape[0] % self.size} point(s) won't be assigned to a cluster!")

        initial_assignment, unassigned_indices = self.initialize(data, n_clusters)
        final_assignment = self.determine_clusters(data, n_clusters, initial_assignment)

        # self.nn_prediction = NearestNeighbors(n_neighbors=prediction_nn, n_jobs=-1).fit(data)
        
        if len(unassigned_indices) > 0:
            
            # it may happen that some points are left out for determining clusters
            # so label them afterwards in order to have a label for each points
            unassigned_data = np.take(data, unassigned_indices, axis=0)
            
            # get distance to each cluster center
            dist = np.repeat(unassigned_data[np.newaxis, :, :], self.centers.shape[0], axis=0) - \
                self.centers.reshape((self.centers.shape[0], 1, data.shape[1]))
            dist = np.abs(np.linalg.norm(dist, axis=2))
            
            # closest cluster are preferred
            preferred = np.argmin(dist, axis=0)
            final_assignment[unassigned_indices] = preferred.flatten()

        super().__init__(data=data, n_clusters=n_clusters, labels=final_assignment, annotations=annotations)

    def soft_predict(self, x: np.ndarray):

        n, _ = x.shape

        data = np.repeat(x[:, :, np.newaxis], self.n_clusters, axis=2)
        dist = np.linalg.norm(data - np.repeat(self.centers.T[np.newaxis, :, :], n, axis=0), axis=1)

        rows_sum = dist.sum(axis=1)
        gamma = 1 - (dist / rows_sum[:, np.newaxis])

        # normalize again
        gamma = gamma / gamma.sum(axis=1)[:, np.newaxis]

        return gamma
    
    def predict(self, x: np.ndarray):

        soft = self.soft_predict(x)
        prediction = np.argmax(soft, axis=1)
        
        if len(prediction) == 1:
            return prediction.item()

        return prediction

if __name__ == "__main__":

    TEST_METHOD = "SameSizeKMeans"

    # random test set for clustering
    points = np.vstack(((np.random.randn(120, 2) * 0.5 + np.array([1, 0])),
        (np.random.randn(100, 2) * 0.25 + np.array([0.5, -1])),
        (np.random.randn(90, 2) * 0.25 + np.array([-0.5, 0.5])),
        (np.random.randn(100, 2) * 0.25 + np.array([-1, -1])),
        (np.random.randn(90, 2) * 0.5 + np.array([-0.5, -0.5]))))
    
    if TEST_METHOD == "SameSizeKMeans":

        skm = SameSizeKMeans(points, n_clusters=4, prediction_nn=1000)

        X = [[] for _i in range(skm.n_clusters)]
        Y = [[] for _i in range(skm.n_clusters)]

        for i in range(points.shape[0]):
            X[skm.labels[i]].append(points[i, 0])
            Y[skm.labels[i]].append(points[i, 1])

        colors = []

        for j in range(skm.n_clusters):
            s = plt.scatter(X[j], Y[j])
            colors.append(s.get_facecolor())

        num_random_points = 50
        x_min, y_min = np.amin(points, axis=0)
        x_max, y_max = np.amax(points, axis=0)

        for k in range(num_random_points):

            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            c = skm.predict(np.asfarray([[x, y]]))

            plt.scatter([x], [y], edgecolors="black", marker="^", c=colors[c].flatten().tolist())

        plt.show()
