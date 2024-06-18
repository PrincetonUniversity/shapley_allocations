import argparse
import time
from datetime import timedelta
import pandas as pd
import numpy as np
from random import seed

from shapley_allocations.shapley.shapley import calc_shapley_var
from shapley_allocations.shapley.char_funcs import output_rate, output_carbon_excess, output_carbon_price
#from shapley.shapley import calc_shapley_var
#from shapley.char_funcs import output_rate, output_carbon_excess, output_carbon_price



POSSIBLE_CHAR_FUNCS = ["rate", "carbon_excess", "carbon_price"]
def select_char_func(inp: str) -> callable:
    if inp=="rate":
        return output_rate
    elif inp=="carbon_excess":
        return output_carbon_excess
    elif inp=="carbon_price":
        return output_carbon_price
    else:
        raise ValueError(f"Unknown characteristic function: only {POSSIBLE_CHAR_FUNCS} are "
                         "allowed.")

if __name__=="__main__":
    #Start the timer
    starttime = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, help="Path to the input file.")
    parser.add_argument("--area_fname", type=str, help="Path to the area filename csv file.")
    parser.add_argument("--output_file", type=str, help="Path to the desired output file.")
    parser.add_argument("--num_cohorts", type=int, help="Number of cohorts.")
    parser.add_argument("--alpha", type=float, help="1-quantile.", default=0.05)
    parser.add_argument("--num_scen", type=int, help="Number of scenarios.", default=500)
    parser.add_argument("--group_cols", type=str, help="Index columns to be used, separated by ,", default="Scenario,Hour")
    parser.add_argument("--output_cols", type=str, help="The numeric data to be used, separatred by ,", 
                        default="CO2 Emissions metric ton,Dispatch,Unit Cost")
    parser.add_argument("--asset_id", type=str, help="In the file used for cohorting, "
                        "unique id of asset", default="GEN UID")
    parser.add_argument("--sec_asset_id", type=str, help="In the main file, "
                        "unique id of asset column", default="Generator")
    parser.add_argument("--cluster_fname", type=str, help="Path to the cluster file, if provided ", 
                        default="")
    parser.add_argument("--char_func", type=str, help="Characteristic function that is used to "
                        "calculate the each coalition's characteristic value",
                        choices=POSSIBLE_CHAR_FUNCS)
    parser.add_argument("--allowance", type=float, help="Carbone allowence for each generator. This value is used only if char_func is carbon_excess or carbon_price.", default=50.)
    parser.add_argument("--is_absolute", dest="is_absolute", action="store_true", help="Whether the carbon allownce is relative (default) or absolute (if this is passed). This value is used only if char_func is carbon_excess or carbon_price.")
    parser.set_defaults(is_absolute=False)
    parser.add_argument("--price_fname", type=str, help="Path to a .csv file containing the LMP prices. This value is used only if char_func is carbon_price.", default="")
    
    args = parser.parse_args()

    folder_path = args.folder_path
    area_fname = args.area_fname
    output_file = args.output_file
    num_cohorts = args.num_cohorts
    alpha = args.alpha#the percentile used for computing the CVaR
    num_scen = args.num_scen
    group_cols = args.group_cols.split(",")
    output_cols = args.output_cols.split(",")
    asset_id = args.asset_id
    sec_asset_id = args.sec_asset_id
    cluster_fname = args.cluster_fname
    char_func = select_char_func(args.char_func)
    allowance = args.allowance
    is_absolute = args.is_absolute
    price = pd.read_csv(args.price_fname) if args.price_fname else None
    

    calc_shapley_var(folder_path=folder_path, output_cols=output_cols, group_cols=group_cols,
                     sec_asset_id=sec_asset_id, alpha=alpha, output=char_func,
                     cluster_fname=cluster_fname, num_cohorts=num_cohorts, area_fname=area_fname, asset_id=asset_id,
                     num_scen=num_scen, output_file=output_file, allowance=allowance,
                     is_absolute=is_absolute, price=price)

    duration = timedelta(seconds=time.perf_counter()-starttime)
    print('Job took: ', duration)
