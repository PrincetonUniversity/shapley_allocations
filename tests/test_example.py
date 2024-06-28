import unittest
import pandas as pd
import numpy as np

from shapley_allocations.shapley.char_funcs import output_rate
from shapley_allocations.shapley.char_funcs import output_carbon_excess
from shapley_allocations.shapley.char_funcs import output_carbon_price
#from shapley.char_funcs import output_rate

class TestExample(unittest.TestCase):
    #Note that every test starts with the prefix "test_".
    def setUp(self) -> None:
        """
        This function is called before every individual test.
        """

        #Initialize the class with my variables
        self.num_scen = 5
        self.num_cohorts = 2
        #self.folder_path = "../tests/scenarios"
        #self.area_fname = "../tests/test_generator_id.csv"
        self.asset_id = "GEN UID"
        self.group_cols = ["Scenario", "Hour"]
        self.output_cols = ["CO2 Emissions metric ton", "Dispatch","Unit Cost"]
        self.sec_asset_id = "Generator"
        self.index = [range(self.num_scen), [0,1]* 2]
        self.df = pd.DataFrame(data = np.transpose(np.array([list(range(1,21)), [1] * 20, 
                                                             list(range(20,0,-1)), [1,1,0,0] * 5],
                                                               dtype= int)),
                               columns = self.output_cols + [self.sec_asset_id],
                               index = pd.MultiIndex.from_product(self.index, names = self.group_cols))
        self.allowance = 10
        self.price = pd.Series(data = [20]* 2, name = "price")

        #Vairables for the shap_alloc function
        self.alpha = 0.1
    
    def test_output_rate(self):
        expected_index = [range(self.num_scen), [0,1]]
        expected_result = pd.Series(data = np.array([2, 3, 6, 7, 10, 11, 14, 15, 18, 19], dtype= int),
                               index = pd.MultiIndex.from_product(expected_index, names = self.group_cols), 
                               name = "Output")
        result = output_rate(self.df.groupby(by = self.group_cols), self.output_cols)
        self.assertListEqual(result.to_list(), expected_result.to_list())

    def test_output_carbon_excess(self):
        expected_index = [range(self.num_scen), [0,1]]
        expected_result = pd.Series(data = np.array([0, 0,0, 0, 0, 2, 8, 10, 16, 18]),
                               index = pd.MultiIndex.from_product(expected_index, names = self.group_cols), 
                               name = "Output")
        result = output_carbon_excess(self.df.groupby(by = self.group_cols), self.output_cols, self.sec_asset_id,
                                    self.allowance, True, False)
        self.assertListEqual(result.to_list(), expected_result.to_list())                      

    def test_output_carbon_price(self):
        expected_index = [range(self.num_scen), [0,1]]
        expected_unit_cost = pd.Series(data = np.array([19, 18,15, 14, 11, 10, 7, 6, 3, 2]),
                               index = pd.MultiIndex.from_product(expected_index, names = self.group_cols), 
                               name = "Output")
        expected_profit = pd.Series(data = np.array([1, 2,5, 6, 9, 10, 13, 14, 17, 18]),
                               index = pd.MultiIndex.from_product(expected_index, names = self.group_cols), 
                               name = "Output")
        expected_excess = pd.Series(data = np.array([0, 0,0, 0, 0, 2, 8, 10, 16, 18]),
                               index = pd.MultiIndex.from_product(expected_index, names = self.group_cols), 
                               name = "Output")
        expected_intensity = pd.Series(data = np.array([2, 3, 6, 7, 10, 11, 14, 15, 18, 19]),
                               index = pd.MultiIndex.from_product(expected_index, names = self.group_cols), 
                               name = "Output")
        expected_result = pd.Series(data = np.array([0, 0,0, 0, 0,10*(2/11), 13*(8/14), 14*(10/15), 17*(16/18), 18*(18/19)]),
                               index = pd.MultiIndex.from_product(expected_index, names = self.group_cols), 
                               name = "Output")
        
        result = output_carbon_price(self.df.groupby(by = self.group_cols), self.output_cols, self.sec_asset_id,
                                     self.allowance, True, False, self.price)
        self.assertListEqual(result.to_list(), expected_result.to_list()) 
    
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

temp = TestExample()
temp.setUp()
print(temp.test_output_carbon_price())
