import argparse
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from random import seed

from shapley.shapley import tail_sims, gen_cohort, gen_cohort_payoff, shap, calc_col, cluster_values, var_shap
from utils.utils import cluster_from_cat, scen_file_list, scen_import, output_rate, output_carbon_excess, output_carbon_price
#Start the timer
starttime = time.perf_counter()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, help="Path to the input file.")
    parser.add_argument("--area_fname", type=str, help="Path to the area filename csv file.")
    parser.add_argument("--output_file", type=str, help="Path to the desired output file.")
    parser.add_argument("--num_cohorts", type=int, help="Number of cohorts.")
    parser.add_argument("--alpha", type=float, help="1-quantile.", default=0.05)
    parser.add_argument("--num_scen", type=int, help="Number of scenarios.", default=500)
    parser.add_argument("--group_cols", type=list, help="Index columns to be used", 
                        default=["Scenario", "Hour"])
    parser.add_argument("--output_cols", type=list, help="The numeric data to be used", 
                        default=["CO2 Emissions metric ton", "Dispatch", "Unit Cost"])
    parser.add_argument("--asset_id", type=str, help="In the file used for cohorting, "
                        "unique id of asset", default="GEN UID")
    parser.add_argument("--sec_asset_id", type=str, help="In the main file, "
                        "unique id of asset column", default="Generator")
    parser.add_argument("--cluster_fname", type=str, help="Path to the cluster file, if provided ", 
                        default="")
    parser.add_argument("--output_kwargs", type=str, help="Extra parameters needed for output functions")
    
    args = parser.parse_args()

    #Set random seed
    seed(1)
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

    group_cols = args.group_cols

    output_cols = args.output_cols

    asset_id = args.asset_id

    sec_asset_id = args.sec_asset_id

    cluster_fname = args.cluster_fname

    output_kwargs = eval(args.output_kwargs)
    price_df = pd.Series(data = [60] * 24, name = "Price")
    output_kwargs["price_df"] = price_df

    output = output_carbon_price
    #output = output_carbon_excess

    #Obtain the path of all the files
    files_list, num_files = scen_file_list(folder_path)
    #FIle size is not too large so read all of it at once and leave it in memory stack
    scen_em_df = scen_import(files_list,output_cols + [group_cols[1]] + [sec_asset_id], [5,9])

    #If there are optional arguments, list them here
    #a = kwargs.get("allowance", None)

    #Obtain the worst alpha% of scenario instances for each hour
    #This provides a list of all the worst instances
    tail_scen = tail_sims(alpha, calc_col(scen_em_df, output_cols, sec_asset_id, group_cols, 
                                          output, **output_kwargs), group_cols)
    #In order to locate the worst scenarios, we filter through the indices
    scen_em_df.set_index(group_cols, inplace= True)

    #Filter for only the tail scenarios
    scen_em_df = scen_em_df.loc[tail_scen,:]

    if (cluster_fname):
        #If cluster file provided
        cohort_df = cluster_from_cat(pd.read_csv(cluster_fname,
                                                 header = 0, index_col = 0))
    else:
        #Generate my own cohorts
        cohort_df = gen_cohort(num_cohorts, area_fname, asset_id)
        
    group_cluster_df = cluster_values(scen_em_df, sec_asset_id, cohort_df, group_cols,
                                  output_cols, ["sum"] * len(output_cols))
    del scen_em_df
    
    char_matrix = gen_cohort_payoff(group_cluster_df, num_cohorts, 
                                    group_cols, output_cols, sec_asset_id, output, **output_kwargs)
    #Append the grand coalition characteristic to the final column
    result = var_shap(np.append(shap(char_matrix, num_cohorts), 
                                char_matrix[:,0].reshape(len(tail_scen),1), axis = 1),
                      tail_scen, group_cols, np.ceil(alpha*num_scen))
    pd.DataFrame(round(result,3)).to_csv(output_file)

duration = timedelta(seconds=time.perf_counter()-starttime)
print('Job took: ', duration)
