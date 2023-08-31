import unittest
from vectorim import Vectorim

EMBEDDING_DATA = [
    "We do not aviate in this house",
    "The largest cat known is the liger",
    "I sure hope dixie's never closes",
]

EMBEDDING_VECTORS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

SEARCH_VECTOR = [0.2, 0.8, 0.2]

EXPECTED_RESULT = "The largest cat known is the liger"


class TestSearch(unittest.TestCase):
    def test_search(self):
        # create a database
        vector_db = Vectorim(vectors=EMBEDDING_VECTORS, data=EMBEDDING_DATA)
        # search for a vector
        _, top_data = vector_db.search(SEARCH_VECTOR, top_k=1)
        result = top_data[0]
        # check the result
        self.assertEqual(result, EXPECTED_RESULT)


unittest.main()
