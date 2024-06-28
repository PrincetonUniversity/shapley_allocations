from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from random import shuffle
from shapley_allocations.utils.utils import cluster_from_cat, scen_file_list, scen_import, split_indx, char_order, \
                                    to_bool_list, shap_Phi
#from utils.utils import cluster_from_cat, scen_file_list, scen_import, split_indx, char_order, \
 #                                   to_bool_list, shap_Phi

#################################
#PREPROCESSING: TAILORED FOR THE SPECIFIC INPUT
#################################
def calc_col(df: pd.DataFrame, output_cols: list[str], player_id_col: str, group_cols: list[str], 
             output_fctn: Callable, **kwargs) -> pd.DataFrame:
    """Generates the new calculated column from the output cols and function. 
    
    Parameters
    -----------
    folder_path: str
        the name of the folder path containing the scenario files
    output_cols: list
        the needed columns to calculate the new column
    player_id_col: str

    group_cols: list
        the groupby column if needed
    output_fctn: function
        user inputted function to do the calculation
    **kwargs:
        other arguments of the output function

    Returns
    -------
    Dataframe
        A dataframe with the newly generated output columns index by the group by
    """
    #Aggregate to the level given then input that into the arbitrary function
    computed_df = output_fctn(df.groupby(by = group_cols), output_cols, player_id_col, 
                              kwargs.get("allowance", None), kwargs.get("is_absolute", None), False,
                              kwargs.get("price_df", None))
    #Change all NaNs to zero
    computed_df.fillna(0, inplace = True)

    return(computed_df)

def gen_cohort(num_cohorts: int, area_fname: str, gen_id_col: str) -> pd.Series:
    """Given the number of clusters desired, and a file name specifying
    each player, generates the clusters for testing the speed of shapley allocations
    
    Parameters
    -----------
    num_cohorts: int
        the number of cohorts to be generated from the files given. The cohorts
        will be generated by generator name
    area_fname: str
        the name of the file containing the area mapping corresponding
        to the generator name
    gen_id_col: str
        the column name that contains the id of the generators

    Returns
    -------
    pd.Series
        Returns the clustering dictionary and also saves it as a csv
    """
    #Open up the areas file 
    gen_zone_df = pd.read_table(area_fname, sep = ",", header = 0, 
                                index_col = gen_id_col)

    #Generate the cohort list
    cohort_indx = list(split_indx(len(gen_zone_df.index.values),num_cohorts))
    shuffle(cohort_indx)

    #Create a column in gen_zone_df that shows which cluster the cohort is in
    gen_zone_df = gen_zone_df.assign(Cluster = cohort_indx)
    #Save to a csv
    pd.DataFrame(gen_zone_df["Cluster"]).to_csv("cohort.csv")
    return(gen_zone_df["Cluster"])

def gen_cluster_values(scen_em_df: pd.DataFrame, player_id_col: str,
                   cohorts_df: pd.Series, group_cols: list[str],
                   output_cols: list[str], group_fctn: list[Callable]) -> pd.DataFrame:
    """Given the dataframe with appropriate cols, the player id column,
    the df containing the labels of the cluster for each player, 
    the index values of the df (could be multi-index), 
    the columns used for computation, and the method of aggregating the clusters
    create the characteristic vector for each tail scenario
    
    Parameters
    -----------
    scen_em_df: pd.DataFrame
        A dataframe containing the values that we want to compute the characteristic
        values. Usually the output of the scen_import function
    player_id_col: str
        the column name of each player in the combined scenario data. This is required
        for identifyling the player's cluster
    cohorts_df: dataframe
        dataframe with unique ID index and their corresponding cluster
    group_cols: list
        the groupby columns, which index scen_em_df if needed
    output_cols: list
        a list containing the column names to perform computation on
    group_fctn: Callable
        function to apply to clustering aggregation. Usually the sum


    Returns
    -------
    pd.DataFrame
        The newly groupled dataframe including an aggregation on the clusters
    """

    #Join the cohort df with the scen data frame
    scen_em_df = pd.merge(left = scen_em_df, right = cohorts_df, how = "left", 
                          left_on= player_id_col, right_index=True)


    #When aggregating to cluster level, specify how they should be aggregated
    computed_df = scen_em_df.groupby(by = group_cols + [cohorts_df.name]).agg(
        dict(zip(output_cols + [player_id_col], group_fctn + ["count"]))
    ).set_axis(output_cols + [player_id_col], axis = 1)

    return(computed_df)

def gen_cohort_payoff(group_cluster_df: pd.DataFrame, num_cohorts: int, 
                      group_cols: list[str], output_cols: list[str], player_id_col: str, output_fctn: Callable,
                      **kwargs) -> np.ndarray:
    """Given the raw dataframe with appropriate cols, the number of clusters, 
    the respective df containing the labels of the clusters, 
    the column name containing the IDs of the clusters,
    the identifying column of the data, the columns used for computation, 
    create the characteristic vector for each tail scenario
    
    Parameters
    -----------
    scen_em_df: pd.DataFrame
        A dataframe containing the values that we want to compute the characteristic
        values. Usually the output of the scen_import function
    player_id_col: str
        the column name of each player in the combined scenario data. This is required
        for identifyling the player's cluster
    num_cohorts: int
        the number of clusters that we eventually perform shapley allocations on
    cohorts_df: dataframe
        dataframe with unique ID index and their corresponding cluster
    group_cols: list
        the groupby columns, which index scen_em_df if needed
    group_fctn: Callable
        function to apply to clustering aggregation. Usually the sum
    output_cols: list
        a list containing the column names to perform computation on
    output_fctn: function
        user inputted function to do the calculation
    **kwargs:
        other arguments of the output function or for groupby


    Returns
    -------
    np.ndarray
        A multi-dim array containing the characteristic functions
    """

    #Create a list containing the 2^n binary representation of cohorts
    #where a 1 indicates that cohort 1 is included
    char_labels = char_order(num_cohorts)

    #Create a list of 2^n that holds the allocations of each cluster
    char_values = [None] * (2**num_cohorts)
    #Obtain the list of different clusters, to be used for slicing the dataframe
    cluster_list = group_cluster_df.index.unique(level= len(group_cols))

    for n in range((2**num_cohorts) - 1):      
        #Loop throught the different combinations of clusters

        #Slice the data for only the required cohorts
        char_cluster_df = group_cluster_df[group_cluster_df.index.get_level_values(len(group_cols)).
                                           isin(cluster_list[to_bool_list(char_labels[n])])]
        #Fill in the list with the outputs needed
        char_values[n] = np.array(output_fctn(char_cluster_df.groupby(group_cols), output_cols, player_id_col,
                              kwargs.get("allowance", None), kwargs.get("is_absolute", None), True,
                              kwargs.get("price_df", None)).values)
    #The characterist function of the empty cluster is always 0
    char_values[(2**num_cohorts) - 1] = np.zeros(np.shape(char_values[0]))
    #Combine all the arrays in list into a matrix
    return(np.stack(char_values, axis = 1))    

#################################
#ANALAYSIS - STAYS THE SAME FOR ALL INPUTS
#################################
def find_simulation_tail(alpha: float, scen_df: pd.Series, grp_by_col: list[str]) -> list:
    """Given the alpha used to compute the CVar, the scenario data series 
    with the outputs to compute the cvar from,
    outputs the tail scenarios 
    
    Parameters
    -----------
    alpha: float
        a float between 0 and 1 that corresponds to how much of the tail
        of the total allocations that we sample for the CVaR computation
    scen_df: obj
        the dataframe generated above that has the output values
    grp_by_col: str
        the column name to group by and use to find the quantiles

    Returns
    -------
    list
        scenarios that are in the tail
    """
    #Define the quantile as the inverted alpha
    quantile = 1 - alpha
    #ALways want to group by the lowest index level
    var_df = scen_df.groupby(level = grp_by_col[-1]).transform("quantile",
                                                               quantile, interpolation = "lower")
    #return hour and their corresponding scenarios
    return(scen_df.loc[scen_df > var_df].index.to_list())

def calc_shapley_value(char_mat: np.ndarray, num_cohorts: int) -> np.ndarray:
    """Given the characteristic matrix and the number of cohorts,
    compute the shapley
    
    Parameters
    -----------
    char_mat: np.ndarray 
        a 3d matrix containing all the characteric values. The Cv in the paper
    num_cohorts: int
        the number of cohorts to be generated from the files given. The cohorts
        will be generated by generator name

    Returns
    -------
    np.ndarray
        A matrix containing the the cohort allocations
    """
    return(np.matmul(char_mat, shap_Phi(num_cohorts)))

def calc_empirical_var(alloc_array: np.ndarray, tail_scen: list, 
             group_cols: list[str], num_tail_scen: int) -> pd.DataFrame:
    """Given the allocation matrix for all the different indices, and the name of the index,
     and number of tail event, compute the empirical var
    
    Parameters
    -----------
    alloc_array: np.ndarray 
        a 2d matrix containing all the allocation from the shapley computation.
    tail_scen: list
        list of index labels for all the tail events
    group_cols: list
        the groupby columns, which index scen_em_df if needed
    num_tail_scen: int
        the number of tail events for normalizing

    Returns
    -------
    Dataframe
        A dataframe containing the the cohort allocations
    """
    mul_index =  pd.MultiIndex.from_tuples(tail_scen, names=group_cols)
    alloc_df = pd.DataFrame(alloc_array, index = mul_index)
    alloc_df = alloc_df.groupby(level = len(group_cols)-1).sum() / num_tail_scen
    return(alloc_df.sort_index())

def calc_shapley_var(folder_path: str, output_cols: list[str], group_cols: list[str],
                     sec_asset_id: str, alpha: float, output: callable,
                     cluster_fname: str, num_cohorts: int, area_fname: str, asset_id: str,
                     num_scen: int, output_file: str, **kwargs):

    #Obtain the path of all the files
    files_list, num_files = scen_file_list(folder_path)
    #FIle size is not too large so read all of it at once and leave it in memory stack
    scen_em_df = scen_import(files_list,output_cols + [group_cols[1]] + [sec_asset_id], [5,9])

    #If there are optional arguments, list them here
    #a = kwargs.get("allowance", None)

    #Obtain the worst alpha% of scenario instances for each hour
    #This provides a list of all the worst instances
    tail_scen = find_simulation_tail(alpha, calc_col(scen_em_df, output_cols, sec_asset_id, group_cols, 
                                          output, **kwargs), group_cols)
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
        
    group_cluster_df = gen_cluster_values(scen_em_df, sec_asset_id, cohort_df, group_cols,
                                  output_cols, ["sum"] * len(output_cols))
    del scen_em_df
    
    char_matrix = gen_cohort_payoff(group_cluster_df, num_cohorts, 
                                    group_cols, output_cols, sec_asset_id, output, **kwargs)
    #Append the grand coalition characteristic to the final column
    result = calc_empirical_var(np.append(calc_shapley_value(char_matrix, num_cohorts), 
                                char_matrix[:,0].reshape(len(tail_scen),1), axis = 1),
                      tail_scen, group_cols, np.ceil(alpha*num_scen))
    pd.DataFrame(round(result,3)).to_csv(output_file, header = ["Coalition " + str(i) for i in range(num_cohorts)] + ["Total Allocation"])