from itertools import compress
from pathlib import Path

import numpy as np
import pandas as pd

# from utils import gen_type, shap_alloc_helper, char_order, to_bool_list
from utils.utils import char_order, gen_type, shap_alloc_helper, to_bool_list


#Compute Shapley Allocations of All Generators
def asset_alloc(num_scen: int, alpha: float, folder_path: str, area_fname: str):
    folder_path = Path(folder_path)
    quantile = 1 - alpha
    #################################
    #PREPROCESSING: TAILORED FOR THE SPECIFIC INPUT
    #################################

    #Open all emission files
    #iterate over directory and store filenames in a list
    #Only keep the csv files and put file names in a list
    files_list = list(filter(lambda f: f.name.endswith('.csv'), folder_path.iterdir()))

    #Create a df to hold the total emissions for each hour and scenario
    full_em_df = pd.DataFrame()
    
    
    for i in range(len(files_list)):
        #For each scenario file, open and read it into a temporary df. Then append it to the 
        #previous data frame
        temp_df = pd.read_table(files_list[i], sep = ",",
                         header = 0, index_col = False,
                          usecols = ["Date", "Hour", "Generator", "CO2 Emissions metric ton", 
                                     "Dispatch"], nrows = 24*3)
        #label the scenario number to keep track
        temp_df["scenario"] = files_list[i].name[5:9]
        full_em_df =  pd.concat([full_em_df, temp_df], ignore_index = True, sort=False)
    
    #Create new column with generator type
    full_em_df["Type"] = full_em_df["Generator"].apply(lambda x: gen_type(x))
    
    del temp_df

    #Open up the areas file 
    #import the data
    gen_zone_df = pd.read_table(area_fname, sep = ",",
                         header = 0, index_col = False)
    #Merge the two data sets
    full_em_df = full_em_df.merge(gen_zone_df, how = "left",
                                  left_on = "Generator", right_on = "GEN UID", copy = False)

    del gen_zone_df

    #Generate the cohort list
    #Cohort is the actual generator 
    #Use dict from keys to create an ordered set
    cohort = list(dict.fromkeys(full_em_df["Generator"]))
    #The cohort size

    n = len(cohort)
    #print(cohort)

    #Create a new vector holding the characteristic function matrix
    char_df = pd.DataFrame({"Hour": full_em_df["Hour"], "Type": full_em_df["Type"], 
                            "Generator": full_em_df["Generator"],
                            "Area": full_em_df["Area"], "Scenario": full_em_df["scenario"]})
    #Which column are we using to determine the cohorts
    category = "Generator"
    #Create the matrix of repeated C02 emissions metric ton
    emission_mat = np.tile(full_em_df["CO2 Emissions metric ton"].values,(2**n,1))
    #Create the matrix of repeated dispatch
    dispatch_mat = np.tile(full_em_df["Dispatch"].values,(2**n,1))
    #Merge these two together
    temp_mat = np.concatenate((emission_mat, dispatch_mat), axis = 0)
    
    #Check to make sure the merge happened correctly
    del emission_mat, dispatch_mat
    
    #Convert into  dataframe
    temp_df = pd.DataFrame(data = np.transpose(temp_mat),
                           columns = [i + "e" for i in char_order(n)] + 
                           [i + "d" for i in char_order(n)], 
                           index = range(len(full_em_df["CO2 Emissions metric ton"].values)))
    
    #Merge the two data frames togetjer
    char_df = pd.concat([char_df, temp_df], axis = 1)
    del temp_df
    
    #Plug in the 0s for idealized cases
    #skip the first one which is the BAU case
    for i in char_order(n)[1:]:
        #Change the value of the new emission column and dispatch column to 0
        char_df.loc[char_df[category].isin(list(compress(cohort,
                                                         to_bool_list(i)))),[i + "e", i + "d"]] = 0
    #Convert the last column back into t

    #Aggregate the data to scenario level to get the contribution values by scenario
    #Group the data frame by scenario and hour. The aggregation only requires summing
    #the allocations and the total emissions
    scen_df = char_df.drop(["Type", "Area", "Generator"],
                           axis = 1).groupby(["Scenario", "Hour"]).sum()
    #Convert the last column back into nonzeros so we dont divide by 0
    scen_df[char_order(n)[-1] + "d"] = scen_df[char_order(n)[0] + "d"]
    
    #Compute the actual payoff value
    scen_df = scen_df.iloc[:,0:(2**n)]/ scen_df.iloc[:,(2**n):(2**(n+1))].values
    #Rename columns
    scen_df.columns = char_order(n)
    #Change all nonporucing assets from Nan to zeros
    scen_df = scen_df.fillna(0)
    
    #################################
    #ANALAYSIS - STAYS THE SAME FOR ALL INPUTS
    #################################
    #First get all the column vectors needed for shapley allowcation
    temp_mat = np.array([shap_alloc_helper(i+1,n) for i in range(n)])
    #Compute the shapley allocations
    scen_df = pd.concat([pd.DataFrame(data = np.matmul(scen_df.values, 
                                            np.transpose(temp_mat)), 
                           columns = cohort, 
                           index = scen_df.index),
                         scen_df], axis =1 )

    #Compute the 95th percentile for each hour accross all scenarious in the BAU case
    scen_df["VaR"] = scen_df[char_order(n)[0]].groupby(level = "Hour").transform("quantile",
                                                                                 quantile)

    #Create a column that indicates if in the tail
    scen_df["inTail"] = scen_df[char_order(n)[0]] >= scen_df["VaR"]

    result_df = scen_df.loc[scen_df["inTail"], cohort + [char_order(n)[0]]]\
                .groupby(level = "Hour").sum() / (num_scen * (1-alpha))
    return(result_df)

