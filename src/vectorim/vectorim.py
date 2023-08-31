"""
VECTORIM
"""
from typing import Any, List
import pickle
import os
import numpy as np


class Vectorim:
    """
    Create, update, save, restore, and search simple vector databases.
    """

    def __init__(
        self,
        vectors: List[List[float]] | np.ndarray = None,
        data: List[Any] = None,
        file_path=None,
    ):
        """
        Create a new vector database. If file_path exists and is valid, a database will be loaded from there.
        In that case DO NOT pass vectors or data (you can appen using Vectorim.append(...)).
        Arguments:
        vectors - Either a list of lists or a numpy matrix of the embedding vectors
        data - A list of data to associate with each vector - ordering should be the same as vectors
        file_path - A path to a file to save the vectors and data to (not required)
        """
        # python safety
        vectors = self._convert_matrix_input(vectors) if vectors else np.ndarray(0)
        data = data if data else []
        # verify inputs
        self._verify_vectors_and_data(vectors, data)
        # initialize state
        self.vectors: np.ndarray = np.ndarray(0)
        self.data = []
        self.file_path = file_path
        if self.file_path:
            try:
                self._load_from_file()
            except OSError as _:
                # this means the file doesn't exist, first run maybe, just keep going
                pass
        # make sure that if we're loading, we don't have init vectors/data
        if np.shape(self.vectors)[0] > 0 and len(vectors) > 0:
            raise ValueError(
                "Can't pass init vectors/data if loading something from file"
            )
        if np.shape(self.vectors)[0] > 0 and len(vectors) == 0:
            # we loaded something, so done init
            return
        # we loaded nothing, take values from init
        self.vectors = vectors
        self.data = data

    def append(
        self,
        vectors: List[List[float]] | np.ndarray,
        data: List,
        save_to_file=True,
    ):
        """
        Append new vectors and data, save to file if needed.
        """
        vectors = self._convert_matrix_input(vectors)
        self._verify_vectors_and_data(vectors, data)
        if np.shape(self.vectors)[0] == 0:
            # if our vectors are empty, just set them
            self.vectors = vectors
        else:
            self.vectors = np.append(self.vectors, vectors, axis=0)
        self.data.extend(data)
        if save_to_file:
            self._save_to_file()

    def search(self, query_vector: List[float] | np.ndarray, top_k=1):
        """
        Search for the top k vectors closest to the query vector.
        """
        query_vector = self._convert_vector_input(query_vector)
        # calculate cosine similarity
        cosine_similarity: np.ndarray = self._cosine_similarity(
            query_vector, self.vectors
        )
        # get top_k indices - top result first
        top_k_indices = np.argsort(cosine_similarity)[-top_k:][::-1]
        # get scores
        top_k_scores = cosine_similarity[top_k_indices].tolist()
        # get data
        top_k_data = [self.data[i] for i in top_k_indices]
        return top_k_scores, top_k_data

    @staticmethod
    def _convert_vector_input(vector) -> np.ndarray:
        """
        Convert list to numpy array if needed
        """
        if isinstance(vector, list):
            # verify it's flat
            if any(isinstance(i, list) for i in vector):
                raise ValueError("Vector input should be a flat list")
            # convert to numpy array
            vector = np.array(vector)
        elif isinstance(vector, np.ndarray):
            # verify it's flat
            if len(np.shape(vector)) > 1:
                raise ValueError("Vector input should be a flat numpy array")
        else:
            raise ValueError("Vector input should be a list or numpy array")
        return vector

    @staticmethod
    def _convert_matrix_input(matrix) -> np.ndarray:
        """
        Convert matrix (list of lists) to numpy array if needed
        """
        if isinstance(matrix, list):
            # verify every entry is a flat list of the same length
            for vector in matrix:
                if any(isinstance(i, list) for i in vector):
                    raise ValueError("Matrix input should be a list of flat lists")
                if len(vector) != len(matrix[0]):
                    print(vector)
                    print("!!")
                    raise ValueError(
                        "Matrix input should be a list of flat lists of the same length"
                    )
            # convert to numpy array
            matrix = np.array(matrix)
        elif isinstance(matrix, np.ndarray):
            # make sure it's 2d
            if len(np.shape(matrix)) != 2:
                raise ValueError("Matrix input should be a 2d numpy array")
        else:
            raise ValueError("Matrix input should be a list or numpy array")
        return matrix

    @staticmethod
    def _verify_vectors_and_data(vectors: np.ndarray, data):
        """
        Simple verification that vectors and data are valid
        """
        # make sure number of vectors (1st dim) is the same as data length
        if np.shape(vectors)[0] != len(data):
            raise ValueError("Vectors and data lists should be of the same length")

    def _save_to_file(self):
        """
        Save vectors and data to file
        """
        if not self.file_path:
            raise ValueError("Called _save_to_file without file path set")
        save_data = {"vectors": self.vectors, "data": self.data}
        with open(self.file_path, "wb") as output_file:
            pickle.dump(save_data, output_file)

    def _load_from_file(self):
        """
        Restore vectors and data from file
        """
        if not self.file_path:
            raise ValueError("Called _load_from_file without file path set")
        # check if path exists
        if not os.path.exists(self.file_path):
            raise OSError("File path does not exist")
        with open(self.file_path, "rb") as input_file:
            load_data = pickle.load(input_file)
        # verify load data
        try:
            self._verify_vectors_and_data(load_data["vectors"], load_data["data"])
        except Exception as exception:
            raise ValueError("Loaded data is invalid") from exception
        # load
        self.vectors = load_data["vectors"]
        self.data = load_data["data"]

    def _cosine_similarity(
        self, vector: List[float] | np.ndarray, matrix: List[List[float]] | np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between a vector and a matrix of possible target vectors
        """
        vector = self._convert_vector_input(vector)
        matrix = self._convert_matrix_input(matrix)
        # calculate dot product
        dot_product = np.dot(matrix, vector)
        # calculate norms
        vector_norm = np.linalg.norm(vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        # calculate cosine similarity
        cosine_similarity = dot_product / (vector_norm * matrix_norms)
        # return as numpy array
        return cosine_similarity
