import argparse
import time
from datetime import timedelta
import pandas as pd
import numpy as np

from shapley.shapley import gen_bau_scen, tail_sims, gen_cohort, gen_cohort_payoff, shap

#Start the timer
starttime = time.perf_counter()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, help="Path to the input file.")
    parser.add_argument("--area_fname", type=str, help="Pass to the area filename csv file.")
    parser.add_argument("--output_file", type=str, help="Path to the desired output file.")
    parser.add_argument("--alpha", type=float, help="1-quantile.", default=0.05)
    parser.add_argument("--num_scen", type=int, help="Number of scenarios.", default=500)
    parser.add_argument("--num_cohorts", type=int, help="Number of cohorts.")
    parser.add_argument("--indx", type=list, help="Index columns to be used", 
                        default=["Scenario", "Hour"])
    parser.add_argument("--output_cols", type=list, help="The numeric data to be used", 
                        default=["CO2 Emissions metric ton", "Dispatch"])
    parser.add_argument("--asset_id", type=str, help="In the file used for cohorting, unique id of asset", 
                        default="GEN UID")
    parser.add_argument("--sec_asset_id", type=str, help="In the main file, unique id of asset column", 
                        default="Generator")
    parser.add_argument("--cluster_fname", type=str, help="Cluster file to use if it already exists", 
                        default="texas7k_cluster.csv")
    args = parser.parse_args()

    #Static variables
    #the percentile used for computing the CVaR
    alpha = args.alpha
    #The number of scenarios
    num_scen = args.num_scen

    #Input files
    folder_path = args.folder_path

    #Filename for the area to generator mapping
    #Open up the areas file 
    area_fname = args.area_fname

    output_file = args.output_file

    num_cohorts = args.num_cohorts

    indx = args.indx

    output_cols = args.output_cols

    asset_id = args.asset_id

    sec_asset_id = args.sec_asset_id

    cluster_fname = args.cluster_fname

    generator_df = gen_bau_scen(folder_path, num_scen, [indx[1]], output_cols)

    tail_scen = tail_sims(alpha, generator_df, indx, "Output")

    cohorts = pd.read_csv(cluster_fname, header = 0, index_col = 0)

    char_matrix = gen_cohort_payoff(num_cohorts, cohorts.squeeze(), folder_path,
                  tail_scen, sec_asset_id, indx, output_cols)
    
    result = shap(char_matrix, num_cohorts, np.ceil(alpha*num_scen))
    pd.DataFrame(result).to_csv(output_file)

duration = timedelta(seconds=time.perf_counter()-starttime)
print('Job took: ', duration)
