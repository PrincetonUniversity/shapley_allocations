import unittest
import pandas as pd

from shapley_allocations.shapley.shapley import gen_cohort

class TestExample(unittest.TestCase):
    #Note that every test starts with the prefix "test_".
    def setUp(self) -> None:
        """
        This function is called before every individual test.
        """

        #Initialize the class with my variables
        self.num_scen = 5
        self.num_cohorts = 3
        self.folder_path = "../tests/scenarios"
        self.area_fname = "../tests/test_generator_id.csv"
        self.asset_id = "GEN UID"
        self.indx = ["Scenario", "Hour"]
        self.output_cols = ["CO2 Emissions metric ton", "Dispatch"]

        #Vairables for the shap_alloc function
        self.alpha = 0.1

    @unittest.skip("Keep this as a template. TODO: DELETE THIS AFTER FELIX CREATED A NEW TEST")
    def test_gen_cohort(self):
        #Suppose my code generates a+b
        result_df = gen_cohort(self.num_cohorts,self.area_fname, self.asset_id)
        expected_result = pd.read_csv("tests/test_gen_cohort.csv", header = 0, index_col = 0)
        pd.testing.assert_frame_equal(result_df, expected_result)

    @unittest.skip("Keep this as a template. TODO: DELETE THIS AFTER FELIX CREATED A NEW TEST")
    def test_tail_sims(self):
        # scen_df = gen_bau_scen(self.folder_path, self.num_scen, [self.indx[1]],self.output_cols)
        # result = find_simulation_tail(self.alpha, scen_df, self.indx, self.output_cols)
        # expected_result = [(1003,0), (1002,1),(1000,2),(1001,3), (1001,4),
        #           (1003,5),(1000,6),(1003,7),(1001,8),(1003,9),
        #           (1001,10),(1002,11),(1000,12),(1002,13),(1001,14),
        #           (1003,15),(1001,16),(1002,17),(1003,18),(1001,19),
        #           (1001,20),(1001,21),(1001,22),(1002,23)]
        # self.assertCountEqual(result, expected_result)
        pass
