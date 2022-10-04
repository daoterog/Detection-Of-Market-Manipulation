"""
Contains mountain, substractive, k-means and fuzzy c-means clustering algorithms.
"""

import numpy as np

from scipy.stats import multivariate_normal

from unsupervised_learning.metrics import (
    manhattan_distance,
    euclidean_distance,
    cosine_distance,
    mahalanobis_distance,
)

from sklearn.base import BaseEstimator, ClusterMixin


class DistanceMetric:

    """Distance metric class."""

    def __init__(self, distance_metric: str):
        """Initializes the DistanceMetric class.
        Args:
            distance_metric (str): Distance metric.
        """
        self.distance_metric = distance_metric
        if distance_metric == "cosine":
            self.distance_criterion = cosine_distance
        elif distance_metric == "euclidean":
            self.distance_criterion = euclidean_distance
        elif distance_metric == "manhattan":
            self.distance_criterion = manhattan_distance
        elif distance_metric == "mahalanobis":
            self.distance_criterion = mahalanobis_distance
        else:
            raise ValueError(
                "Invalid distance metric. Please choose from: cosine, euclidean,"
                + " manhattan, mahalanobis."
            )

        self.inverse_covariance_matrix = None

    def get_distance(self, feature_matrix: np.ndarray, datapoint: np.ndarray) -> float:
        """Returns the distance between a feature_matrix (or datapoint 1) and a datapoint.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
            datapoint (np.ndarray): Data point.
        Returns
        """
        if self.distance_metric == "mahalanobis":
            if self.inverse_covariance_matrix is None:
                covariance_matrix = np.cov(feature_matrix.T, ddof=1)
                cond_number = np.linalg.cond(covariance_matrix)
                self.inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
                if cond_number > 1e15:
                    print(
                        "The covariance matrix is ill-conditioned. Please try another distance"
                        + " metric."
                    )
            distances = [
                self.distance_criterion(
                    first_datapoint.reshape(1, -1),
                    datapoint,
                    self.inverse_covariance_matrix,
                )
                for first_datapoint in feature_matrix
            ]
            if len(distances) == 1:
                return distances[0]
            return np.array(distances).squeeze()
        else:
            return self.distance_criterion(feature_matrix, datapoint)


class KMeans(BaseEstimator, ClusterMixin):

    """K-Means algorithm"""

    def __init__(
        self,
        number_of_clusters: int = 5,
        distance_metric: str = "euclidean",
        n_iter: int = 1000,
        verbose: bool = False,
    ) -> None:
        """Initializes the KMeans class.
        Args:
            number_of_clusters (int): Number of clusters. Defaults to 5.
            distance_metric (str): Distance metric. Defaults to 'euclidean'.
            max_iters (int): Maximum number of iterations. Defaults to 1000.
            verbose (bool): Verbose. Defaults to False.
        """
        self.number_of_clusters = number_of_clusters
        self.max_iters = n_iter
        self.verbose = verbose
        self.distance_criterion = DistanceMetric(distance_metric).get_distance

        self.n_features_in_ = None
        self.assignments_ = None
        self.centers_ = None

    def _initialize_variables(self, feature_matrix: np.ndarray) -> None:
        """Initializes the variables.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Indicate numbers of features to expect for predict calls
        self.n_features_in_ = feature_matrix.shape[1]
        # Initialize assignments matrix
        self.assignments_ = np.zeros(feature_matrix.shape[0])
        # Initialize k cluster centers with random points
        self.centers_ = np.random.permutation(feature_matrix)[: self.number_of_clusters]

    def _parallel_assign_datapoints(self, idx: int, datapoint: np.ndarray) -> None:
        """Assigns datapoints to clusters.
        Args:
            idx (int): Index of datapoint.
            datapoint (np.ndarray): Datapoint.
        """
        # Get distance to clusters
        distance_to_centers = self.distance_criterion(
            self.centers_, datapoint.reshape(1, -1)
        )
        # Assign data point to closest cluster
        self.assignments_[idx] = np.argmin(distance_to_centers)

    def _assign_datapoints(self, feature_matrix: np.ndarray) -> None:
        """Assigns datapoints to clusters.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        for idx, datapoint in enumerate(feature_matrix):
            # Get distance to clusters
            distance_to_centers = self.distance_criterion(
                self.centers_, datapoint.reshape(1, -1)
            )
            # Assign data point to closest cluster
            self.assignments_[idx] = np.argmin(distance_to_centers)

    def _update_cluster_centers(self, feature_matrix: np.ndarray) -> None:
        """Updates the cluster centers.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        for cluster in range(self.number_of_clusters):
            self.centers_[cluster] = np.mean(
                feature_matrix[self.assignments_ == cluster]
            )

    def _compute_cost(self, feature_matrix: np.ndarray) -> None:
        """Computes the cost function.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        Outputs:
            Prints cost function.
        """
        cost = 0
        for cluster in range(self.number_of_clusters):
            cost += np.sum(
                self.distance_criterion(
                    feature_matrix[self.assignments_ == cluster], self.centers_[cluster]
                )
            )
        if self.verbose:
            print(f"Cost: {cost}")

    def fit(self, feature_matrix: np.ndarray, target: np.ndarray = None) -> None:
        """Fits the KMeans algorithm.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
            target (np.ndarray): Target. Defaults to None. This parameter is not used and is only
                added for it to be compliant with the sklearn API.
        """
        # Initialize variables
        self._initialize_variables(feature_matrix)
        # Repeat for max_iter iterations
        for _ in range(self.max_iters):
            # Assign data points to clusters
            self._assign_datapoints(feature_matrix)
            # Update cluster centers
            self._update_cluster_centers(feature_matrix)
            # Compute cost function
            self._compute_cost(feature_matrix)
        return self

    def predict(self) -> np.ndarray:
        """Returns the assignments matrix.
        Returns:
            np.ndarray: Assignments matrix.
        """
        return self.assignments_


class FuzzCMeans(BaseEstimator, ClusterMixin):

    """Fuzzy C-Means algorithm."""

    def __init__(
        self,
        number_of_clusters: int = 5,
        fuzzines_parameter: int = 2,
        distance_metric: str = "euclidean",
        n_iter: int = 1000,
        verbose: bool = False,
    ) -> None:
        """Initializes the FuzzCMeans class.
        Args:
            number_of_clusters (int): Number of clusters. Defaults to 5.
            fuzzines_parameter (int): Fuzzines parameter. Defaults to 2.
            distance_metric (str): Distance metric. Defaults to 'euclidean'.
            max_iters (int): Maximum number of iterations. Defaults to 1000.
            verbose (bool): Verbose. Defaults to False.
        """
        self.number_of_clusters = number_of_clusters
        self.max_iters = n_iter
        self.verbose = verbose
        self.distance_criterion = DistanceMetric(distance_metric).get_distance
        self.fuzzines_parameter = fuzzines_parameter

        self.n_features_in_ = None
        self.assignments_ = None
        self.centers_ = None

    def _initialize_variables(self, feature_matrix: np.ndarray) -> None:
        """Initializes the variables.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Indicate numbers of features to expect for predict calls
        self.n_features_in_ = feature_matrix.shape[1]
        # Initialize membership matrix and normalize it
        self.assignments_ = np.random.rand(
            feature_matrix.shape[0], self.number_of_clusters
        )
        self.assignments_ = (
            self.assignments_ / np.sum(self.assignments_, axis=1)[:, None]
        )
        # Initialize cluster centers
        self.centers_ = np.zeros((self.number_of_clusters, feature_matrix.shape[1]))

    def _update_cluster_centers(self, feature_matrix: np.ndarray) -> None:
        """Updates the cluster centers.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Elevate membership matrix to fuzzines parameter
        membership_coefficients = np.power(self.assignments_, self.fuzzines_parameter)
        # Update cluster centers
        for i in range(self.number_of_clusters):
            self.centers_[i, :] = np.sum(
                feature_matrix * membership_coefficients[:, i].reshape(-1, 1),
                axis=0,
                keepdims=True,
            ) / np.sum(membership_coefficients[:, i], axis=0, keepdims=True)

    def _update_membership_matrix(self, feature_matrix: np.ndarray) -> None:
        """Updates membership matrix.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        for idx, datapoint in enumerate(feature_matrix):
            # Get distance to clusters
            distance_to_centers = self.distance_criterion(
                self.centers_, datapoint.reshape(1, -1)
            )
            # Elevate to fuzzines parameter power
            sum_centers_inverse_distance = np.sum(
                np.power(1 / distance_to_centers, 2 / (self.fuzzines_parameter - 1))
            )
            # Elevate centers distance to fuzzines parameter power
            elevated_centers_distance = np.power(
                distance_to_centers, 2 / (self.fuzzines_parameter - 1)
            )
            # Update membership matrix
            self.assignments_[idx, :] = (
                1 / (elevated_centers_distance * sum_centers_inverse_distance)
            ).squeeze()

    def _compute_cost(self, feature_matrix: np.ndarray) -> None:
        """Computes the cost function.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        Outputs:
            Prints cost function.
        """
        cost = 0
        for cluster in range(self.number_of_clusters):
            powered_membership_coefficients = np.power(
                self.assignments_[:, cluster].reshape(-1, 1), self.fuzzines_parameter
            ).T
            distance_to_centers = self.distance_criterion(
                feature_matrix, self.centers_[cluster, :].reshape(1, -1)
            )
            cost += np.dot(
                powered_membership_coefficients,
                np.power(distance_to_centers, 2),
            ).item()
        print(f"Cost: {cost}")

    def fit(self, feature_matrix: np.ndarray, target: np.ndarray = None) -> None:
        """Fits the FuzzCMeans algorithm.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
            target (np.ndarray): Target. Defaults to None. This parameter is not used and is only
                added for it to be compliant with the sklearn API.
        """
        # Initialize parameters
        self._initialize_variables(feature_matrix)
        # Repeat for max_iter iterations
        for _ in range(self.max_iters):
            # Get cluster centers
            self._update_cluster_centers(feature_matrix)
            # Update membership matrix
            self._update_membership_matrix(feature_matrix)
            # Compute cost
            if self.verbose:
                self._compute_cost(feature_matrix)
        return self

    def predict(self) -> np.ndarray:
        """Returns the assignments matrix.
        Returns:
            np.ndarray: Assignments matrix.
        """
        return self.assignments_


class MountainClustering(BaseEstimator, ClusterMixin):

    """Mountain clustering algorithm."""

    def __init__(
        self,
        number_of_partitions: int = 10,
        distance_metric: str = "euclidean",
        sigma_squared: float = 1,
        beta_squared: float = 1.5,
    ) -> None:
        """Initializes the MountainClustering class.
        Args:
            number_of_clusters (int): Number of clusters. Defaults to 5.
            number_of_partitions (int): Number of partitions. Defaults to 10.
            distance_metric (str): Distance metric. Defaults to 'euclidean'.
            sigma_squared (float): Sigma squared. Defaults to 1.
            beta_squared (float): Beta squared. Defaults to 1.5.
        """
        self.number_of_partitions = number_of_partitions
        self.sigma_squared = sigma_squared
        self.beta_squared = beta_squared
        self.distance_criterion = DistanceMetric(distance_metric).get_distance

        self.n_features_in_ = None
        self.cluster_centers_ = None
        self.cluster_mountain_functions_ = None
        self.assignments_ = None
        self.mountains_snap_shots_ = None

    def get_mountain_snapshots(self) -> np.ndarray:
        """Returns the mountain snapshots.
        Returns:
            np.ndarray: Mountain snapshots.
        """
        return self.mountains_snap_shots_

    def _initialize_parameters(self, feature_matrix: np.ndarray) -> None:
        """Initializes the parameters.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        self.n_features_in_ = feature_matrix.shape[1]
        self.assignments_ = -np.ones((feature_matrix.shape[0], 1))

    def _create_grid(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Creates a grid of points.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        Returns:
            np.ndarray: Grid of points.
        """
        # Get min and max values for each feature
        mins = np.min(feature_matrix, axis=0)
        maxs = np.max(feature_matrix, axis=0)
        # Create grid
        grid = [[]]
        for i in range(feature_matrix.shape[1]):
            new_grid = np.linspace(mins[i], maxs[i], self.number_of_partitions).tolist()
            grid = [
                combination + [new_combination]
                for combination in grid
                for new_combination in new_grid
            ]
        return np.array(grid)

    def _get_mountains(
        self, prototype: np.ndarray, feature_matrix: np.ndarray
    ) -> float:
        """Returns mountain function of certain prototype.
        Args:
            prototype (np.ndarray): Prototype.
            feature_matrix (np.ndarray): Feature matrix.
        Returns:
            np.ndarray: Mountain function.
        """
        squared_distance_to_datapoints = np.power(
            self.distance_criterion(feature_matrix, prototype), 2
        )
        exponential_exponent = -squared_distance_to_datapoints / (
            2 * self.sigma_squared
        )
        return np.sum(np.exp(exponential_exponent))

    def _update_mountains(
        self, prototype: np.ndarray, mountain_function: float
    ) -> float:
        """Updates mountain function of certain prototype.
        Args:
            prototype (np.ndarray): Prototype.
            mountain_function (float): Mountain function.
        Returns:
            float: Updated mountain function.
        """
        scaling_factor = (
            np.exp(
                -np.power(
                    self.distance_criterion(prototype, self.cluster_centers_[-1]),
                    2,
                )
                / (2 * self.beta_squared)
            )
            .squeeze()
            .item()
        )
        return (
            mountain_function
            - self.cluster_mountain_functions_[-1].item() * scaling_factor
        )

    def _update_centers(
        self, mountain_functions: np.ndarray, prototypes: np.ndarray
    ) -> None:
        """Updates cluster centers.
        Args:
            mountain_functions (np.ndarray): Mountain functions.
            prototypes (np.ndarray): Prototypes.
        """
        maximum_index = np.argmax(mountain_functions)
        if self.cluster_centers_ is None:
            self.cluster_centers_ = np.array(prototypes[maximum_index, :]).reshape(
                1, -1
            )
            self.cluster_mountain_functions_ = np.array(
                mountain_functions[maximum_index]
            ).reshape(1, -1)
            self.mountains_snap_shots_ = [mountain_functions]
        else:
            self.cluster_centers_ = np.vstack(
                [self.cluster_centers_, prototypes[maximum_index, :]]
            )
            self.cluster_mountain_functions_ = np.append(
                self.cluster_mountain_functions_, [mountain_functions[maximum_index]]
            )
            self.mountains_snap_shots_.append(mountain_functions)

    def _assign_datapoints(self, feature_matrix: np.ndarray) -> None:
        """Assigns datapoints to clusters.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        self.assignments_ = np.zeros((feature_matrix.shape[0], 1))
        for idx, datapoint in enumerate(feature_matrix):
            # Get distance to clusters
            distance_to_centers = self.distance_criterion(
                self.cluster_centers_, datapoint
            )
            # Update membership matrix
            self.assignments_[idx, 0] = np.argmin(distance_to_centers)

    def fit(self, feature_matrix: np.ndarray, target: np.ndarray = None) -> None:
        """Fits the algithm to the data.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
            target (np.ndarray): Target. Defaults to None. This parameter is not used and is only
                added for it to be compliant with the sklearn API.
        """
        # Initialize parameters
        self._initialize_parameters(feature_matrix)
        # Create prototype grid
        prototypes = self._create_grid(feature_matrix)
        # Evaluate prototypes mountain functions
        mountain_functions = np.array([
            self._get_mountains(prototype.reshape(1, -1), feature_matrix)
            for prototype in prototypes
        ])
        # Get first cluster by finding the maximum of the mountain function
        self._update_centers(mountain_functions, prototypes)
        # Repeat number_of_clusters
        iteration = 0
        while len(mountain_functions):
            # Update mountain function
            mountain_functions = np.array([
                self._update_mountains(prototype.reshape(1, -1), mountain_function)
                for prototype, mountain_function in zip(prototypes, mountain_functions)
            ])
            # Get next cluster by finding the maximum of the mountain function
            self._update_centers(mountain_functions, prototypes)
            # Filterout prototypes with negative densities
            mountain_functions = mountain_functions[mountain_functions > 0]
            iteration += 1
        # Assign datapoints to clusters
        self._assign_datapoints(feature_matrix)
        return self

    def predict(self) -> np.ndarray:
        """Returns the assignments matrix.
        Returns:
            np.ndarray: Assignments matrix.
        """
        return self.assignments_


class SubstractiveClustering(BaseEstimator, ClusterMixin):

    """Substractive clustering algorithm."""

    def __init__(
        self,
        r_a: float = 1.0,
        r_b: float = 1.5,
        distance_metric: str = "euclidean",
    ) -> None:
        """Initializes the SubstractiveClustering class.
        Args:
            number_of_clusters (int): Number of clusters.
        """
        self.r_a = r_a
        self.r_b = r_b
        self.distance_criterion = DistanceMetric(distance_metric).get_distance

        self.n_features_in_ = None
        self.cluster_centers_ = None
        self.centers_densities_ = None
        self.assignments_ = None
        self.mountains_snap_shots_ = None

    def get_mountain_snapshots(self) -> np.ndarray:
        """Returns the mountain snapshots.
        Returns:
            np.ndarray: Mountain snapshots.
        """
        return self.mountains_snap_shots_

    def _initialize_parameters(self, feature_matrix: np.ndarray) -> None:
        """Initializes the parameters.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        self.n_features_in_ = feature_matrix.shape[1]
        self.assignments_ = -np.ones((feature_matrix.shape[0], 1))

    def _get_density(self, prototype: np.ndarray, feature_matrix: np.ndarray) -> float:
        """Returns the density of a datapoint.
        Args:
            datapoint (np.ndarray): Datapoint.
            feature_matrix (np.ndarray): Feature matrix.
        Returns:
            float: Density.
        """
        # Get distance to datapoints
        squared_distance = np.power(
            self.distance_criterion(feature_matrix, prototype), 2
        )
        return np.sum(np.exp(-squared_distance * 4 / (self.r_a**2)))

    def _update_centers(
        self, feature_matrix: np.ndarray, densities: np.ndarray
    ) -> None:
        """Updates cluster centers.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
            densities (np.ndarray): Densities.
        """
        maximum_index = np.argmax(densities)
        if self.cluster_centers_ is None:
            self.cluster_centers_ = np.array(feature_matrix[maximum_index, :]).reshape(
                1, -1
            )
            self.centers_densities_ = np.array(densities[maximum_index]).reshape(1, -1)
            self.mountains_snap_shots_ = [densities]
        else:
            self.cluster_centers_ = np.vstack(
                [self.cluster_centers_, feature_matrix[maximum_index, :]]
            )
            self.centers_densities_ = np.append(
                self.centers_densities_, [densities[maximum_index]]
            )
            self.mountains_snap_shots_.append(densities)

    def _update_densities(self, prototype: np.ndarray, density: np.ndarray) -> None:
        """Returns updated density.
        Args:
            prototype (np.ndarray): Prototype.
            density (np.ndarray): Density.
        Returns:
            float: Updated density.
        """
        scaling_factor = (
            np.exp(
                -np.power(
                    self.distance_criterion(
                        prototype, self.cluster_centers_[-1].reshape(1, -1)
                    ),
                    2,
                )
                * 4
                / (self.r_b**2)
            )
            .squeeze()
            .item()
        )
        return density - self.centers_densities_[-1].item() * scaling_factor

    def _assign_datapoints(
        self, cluster: int, densities: np.ndarray, feature_matrix_index: np.ndarray
    ) -> None:
        """Assigns datapoints with negative densities to a cluster
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        not_assigned_with_negative_density = np.logical_and(
            self.assignments_[feature_matrix_index].squeeze() == -1, densities < 0
        )
        assignment_index = feature_matrix_index[not_assigned_with_negative_density]
        self.assignments_[assignment_index] = cluster

    def fit(self, feature_matrix: np.ndarray, target: np.ndarray = None) -> None:
        """Fits the algorithm to the data.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
            target (np.ndarray): Target. Defaults to None. This parameter is not used and is only
                added for it to be compliant with the sklearn API.
        """
        # Initialize parameters
        self._initialize_parameters(feature_matrix)
        # Index of feature_matrix
        feature_matrix_index = np.arange(feature_matrix.shape[0])
        # Get density of datapoints
        densities = np.array(
            [
                self._get_density(datapoint.reshape(1, -1), feature_matrix)
                for datapoint in feature_matrix
            ]
        )
        # Get first center with highest density
        self._update_centers(feature_matrix, densities)
        # Repeat until every cluster center has been found
        iteration = 0
        while len(densities[densities > 0]):
            # Update densities
            densities = np.array(
                [
                    self._update_densities(datapoint.reshape(1, -1), density)
                    for datapoint, density in zip(feature_matrix, densities)
                ]
            )
            # Assign datapoints to clusters
            self._assign_datapoints(iteration, densities, feature_matrix_index)
            # Get next center with highest density
            self._update_centers(feature_matrix, densities)
            # Filterout datapoints that have been assigned
            not_assigned = self.assignments_[feature_matrix_index].squeeze() == -1
            densities = densities[not_assigned]
            feature_matrix = feature_matrix[not_assigned]
            feature_matrix_index = feature_matrix_index[not_assigned]
            iteration += 1
        return self

    def predict(self) -> np.ndarray:
        """Returns the assignments matrix.
        Returns:
            np.ndarray: Assignments matrix.
        """
        return self.assignments_


class GaussianMixture(BaseEstimator, ClusterMixin):

    """Expectation Maximization clustering algorithm"""

    def __init__(
        self,
        number_of_clusters: int = 5,
        n_iter: int = 1000,
        verbose: bool = False,
    ) -> None:
        """Initializes the ExpectationMaximization class.
        Args:
            number_of_clusters (int): Number of clusters.
        """
        self.number_of_clusters = number_of_clusters
        self.n_iter = n_iter
        self.verbose = verbose

        self.n_features_in_ = None
        self.cluster_centers_ = None
        self.cluster_variances_ = None
        self.probabilities_ = None

    def _initialize_parameters(self, feature_matrix: np.ndarray) -> None:
        """Randomly initializes cluster parameters.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        self.n_features_in_ = feature_matrix.shape[1]
        self.cluster_centers_ = np.random.permutation(feature_matrix)[
            : self.number_of_clusters
        ]
        self.cluster_variances_ = [
            np.eye(self.n_features_in_) for _ in range(self.number_of_clusters)
        ]

    def _compute_probabilities(self, feature_matrix: np.ndarray) -> None:
        """Computes probabilities of each datapoint belonging to each cluster.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Compute distance to clusters
        probs = np.array(
            [
                [
                    multivariate_normal.pdf(
                        datapoint,
                        mean=cluster_center,
                        cov=cluster_variance,
                    )
                    for cluster_center, cluster_variance in zip(
                        self.cluster_centers_, self.cluster_variances_
                    )
                ]
                for datapoint in feature_matrix
            ]
        ).squeeze()
        self.probabilities_ = probs / np.sum(probs, axis=1, keepdims=True)

    def _update_parameters(self, feature_matrix: np.ndarray) -> None:
        """Updates cluster parameters.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
        """
        # Update cluster centers by weighted sum of datapoints and probabilities
        self.cluster_centers_ = (
            np.dot(self.probabilities_.T, feature_matrix)
            / np.sum(self.probabilities_, axis=0, keepdims=True).T
        )
        # Update cluster variances by weighted sum of squared errors and probabilities
        for cluster, center in enumerate(self.cluster_centers_):
            centered_feature_matrix = feature_matrix - center.reshape(1, -1)
            self.cluster_variances_[cluster] = (
                np.dot(centered_feature_matrix.T, centered_feature_matrix)
                / feature_matrix.shape[0]
            )

    def fit(self, feature_matrix: np.ndarray, target: np.ndarray = None) -> None:
        """Fits the algorithm to the data.
        Args:
            feature_matrix (np.ndarray): Feature matrix.
            target (np.ndarray): Target. Defaults to None. This parameter is not used and is only
                added for it to be compliant with the sklearn API.
        """
        # Randomly initialize cluster parameters
        self._initialize_parameters(feature_matrix)
        # Repeat for n_iter times
        for _ in range(self.n_iter):
            # Compute probability that each datapoint belongs to each cluster
            self._compute_probabilities(feature_matrix)
            # Update cluster parameters
            self._update_parameters(feature_matrix)
        return self

    def predict(self) -> np.ndarray:
        """Returns the assignments matrix.
        Returns:
            np.ndarray: Probability matrix.
        """
        return self.probabilities_
