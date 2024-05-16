import unittest

import pandas as pd

from shapley.shapley import gen_cohort, shap
from utils.utils import shap_Phi

class TestExample(unittest.TestCase):
    #Note that every test starts with the prefix "test_".
    def setUp(self) -> None:
        """
        This function is called before every individual test.
        """

        #Initialize the class with my variables
        #The variables for the gen_cohort function
        self.num_scen = 5
        self.num_cohorts = 3
        self.folder_path = "/Users/felix/Programs/orfeus/2018-08-01/emission"
        self.area_fname = "../texas7k_2020_gen_to_zone.csv"
        self.asset_id = "GEN UID"

        #Vairables for the shap_alloc function
        self.alpha = 1/3

    def test_gen_cohort(self):
        #Suppose my code generates a+b
        gen_cohort(self.num_cohorts,self.area_fname, self.asset_id)
        result_df = pd.read_csv("cohort.csv", header = 0, index_col=0)
        expected_result = pd.read_csv("tests/test_gen_cohort.csv", header = 0, index_col = 0)
        pd.testing.assert_frame_equal(result_df, expected_result)

    def test_shap(self):
        phi = shap_Phi(self.num_cohorts)
    '''
    def test_shap_alloc(self):
        #Suppose my code approximates a-b and I know the test should be positive for some reason
        result_df = shap_alloc(self.num_scen, self.alpha, self.num_cohorts, 
                               pd.read_csv("tests/test_gen_cohort.csv", header = [0], index_col = [0,1]))
        expected_result = pd.read_csv("tests/test_shap_alloc.csv", header = [0], index_col = [0])
        pd.testing.assert_frame_equal(result_df, expected_result)
    '''