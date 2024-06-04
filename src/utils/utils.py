import math
from pathlib import Path
import numpy as np
import pandas as pd
#from scipy.linalg import block_diag


#List of helper functions
def gen_type(generator: str) -> str:
    """Given a string of the generator name, returns a substring containing the type of generator
    
    Parameters
    -----------
    generator: str
        string of the generator name

    Returns
    -------
    str
        a string of the generator
    """
    return(generator.split("_")[1])

def rec_coef(N: int) -> np.ndarray:
    """Given the number of clusters in the game, identify the vector that holds all the counting
    coeficients in the shapley value computation. This is the eta in Wang et al
    The computation is executed with dynamic programming (more efficient than recursion).
    
    Parameters
    -----------
    N: int
        the number of clusters in the coop game

    Returns
    -------
    np.array
        a matrix of dimention 2^N x 1. It represents eta_k in page 5
    """
    curr_res = np.array([0])
    for curr_n in range(1, N+1):
        curr_res = np.append(curr_res + np.ones(2**(curr_n-1), dtype=int), curr_res)
    return curr_res

def shap_Phi(num_cohorts: int) -> np.ndarray:
    """Given the number of clusters in the game, compute the matrix Phi from Wang to
    multiply by the characteristic value vection
    
    Parameters
    -----------
    num_cohorts: int
        the number of clusters in the coop game

    Returns
    -------
    np.array
        a matrix of dimention 2^N x N. It represents eta_k in page 5
    """
    #Define the vector that the eta vector is derived from
    l_vector = rec_coef(num_cohorts-1).tolist()
    #Define the eta that I have to split into even blocks
    eta =  np.zeros(2**(num_cohorts-1))
    for j in range(2 ** (num_cohorts-1)):
        eta[j] = (math.factorial(l_vector[j]) * 
                   (math.factorial((num_cohorts-1)-l_vector[j])))
    #Now define the phi using the eta
    Phi = np.zeros(( 2**num_cohorts, num_cohorts))
    for m in range(1, num_cohorts+1):
        eta_m = list(split(eta, 2**(m-1)))
        for j in range(0,(2**m), 2):
            #Work on the positive and negative part at same time
            Phi[j*((2**num_cohorts)//(2**m)):(j+1)*((2**num_cohorts)//(2**m)),
                (m-1)] = eta_m[j // 2]
            Phi[(j+1)*((2**num_cohorts)//(2**m)):(j+2)*((2**num_cohorts)//(2**m)),
                (m-1)] = np.multiply(eta_m[j // 2],-1)
    return(Phi * (1/math.factorial(num_cohorts)))   

#Create a list that gives the ordering for the charateristic function
#Turns out it is just the binary numbers in reverse order
def char_order(N: int) -> list:
    """Given the number of clusters n in the coop game, give the correct order of the payoff
    functions in the structure vector C_v of the payoff function. Needs to be in the correct order
    or the matrix multiplication wont work. It turns out it is just binary numbers in reverse order
    padded with 0s. Also key to note the 00...0 means the perfect world scenario and 11...1 is BAU
    
    Parameters
    -----------
    N: int
        the number of clusters in the coop game

    Returns
    -------
    list
        a list of the order of the payoffs in the vector C_v. 
    """
    
    #The format() function formats the input following the Format Specification mini language. 
    
    #The # makes the format include the 0b prefix, and the 010 size formats the output to fit
    #in 10 characters 
    
    #width, with 0 padding; 2 characters for the 0b prefix, the other 8 for the binary digits.
    
    #order =  [None for _ in range(n**2 -1)]
    order = [None] * (2 ** N)
    for i in range(2 ** N):
        order[i] = format((2 ** N) - i -1, '0' + str(N) + 'b')
    return(order)

#Create a function that converts the binary numbers to boolean list
#in opposite ditection because the 0s are the ones I idealize
def to_bool_list(x: str):
    """Given the binary representation of a number as a string, convert all the
    "1" characters to True and "0" characters to False and output it as a list
    
    Parameters
    -----------
    x: str
        string of s and 0s

    Returns
    -------
    list
        boolean list. 
    """
    #Empty list to hold the output
    list = []
   
    for i in x:
         #Go character by character evaluating the boolean
        list.append(bool(int(i)))
    return(list)

def flatten(xss):
    return [x for xs in xss for x in xs]

def split(item_list: list, n: int):
    """Given a list to split into n equal pieces, return the slices of the list
    
    Parameters
    -----------
    item_list: list
        list that you want to split
    n: int
        the number of equal pieces to split the length

    Returns
    -------
    list
        list of list containing the chunks of the list. 
    """
    k, m = divmod(len(item_list), n)
    return(item_list[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def split_indx(item_length: int, n: int):
    """Given a list length to split into n equal pieces, return the indices of the list
    
    Parameters
    -----------
    item_length: int
        length of list that you want to split
    n: int
        the number of equal pieces to split the length

    Returns
    -------
    list
        list of indices containing the chunks of the list. 
    """
    k, m = divmod(item_length, n)
    
    return(flatten([i] * (k + (min(i+1, m)- min(i, m))) for i in range(n)))

def output_emission_rate(df: pd.DataFrame, output_cols: list, *args):
    """Given a dataframe with only the columns that should be modified
    
    Parameters
    -----------
    df: pd.DataFrame
        A dataframe containing the columns that should be evaluated
    output_cols: list
        list of strings containing the column names

    Returns
    -------
    pd.Series
        elementwise division of the two vectors. 
    """ 
    col_sum_df = df.sum()
    col_sum_df["Output"] = col_sum_df[output_cols[0]] / col_sum_df[output_cols[1]]
    return(col_sum_df["Output"])

def output(df: pd.DataFrame, output_cols: list, carbon_cap: float, 
           num_gen: int = None):
    """Given a dataframe with only the columns that should be modified
    
    Parameters
    -----------
    df: pd.DataFrame
        A dataframe containing the columns that should be evaluated
    output_cols: list
        list of strings containing the column names
    carbon_cap: float
        The maximum amount of carbon allowed

    Returns
    -------
    pd.Series
        elementwise subtraction and take the maximum between that and 0. 
    """ 
    if (not num_gen):
        num_gen = df.size()

    #First obtain number of generators in each hour and total emissions
    col_sum_df = df.sum()
    #col_sum_df = df.agg(count=(output_cols[0], "size"), sum=(output_cols[0], 'sum'))
    # Then multiple the count of generators by the cap
    #col_sum_df["cap"] = col_sum_df["count"].mul(carbon_cap)
    # Get the excess emission
    col_sum_df["excess"] = col_sum_df[output_cols[0]] - (carbon_cap * num_gen)
    #Only take the minimum of that value and zero
    col_sum_df["Output"] = col_sum_df["excess"].clip(lower = 0)
    return(col_sum_df["Output"])

def carbon_excess(df: pd.DataFrame, output_cols: list, carbon_cap: float, 
           num_gen: int = None):
    """Given a dataframe with only the columns that should be modified
    
    Parameters
    -----------
    df: pd.DataFrame
        A dataframe containing the columns that should be evaluated
    output_cols: list
        list of strings containing the column names
    carbon_cap: float
        The maximum amount of carbon allowed

    Returns
    -------
    pd.Series
        elementwise subtraction and take the maximum between that and 0. 
    """ 
    if (not num_gen):
        num_gen = df.size()

    #First obtain number of generators in each hour and total emissions
    col_sum_df = df.sum()
    #col_sum_df = df.agg(count=(output_cols[0], "size"), sum=(output_cols[0], 'sum'))
    # Then multiple the count of generators by the cap
    #col_sum_df["cap"] = col_sum_df["count"].mul(carbon_cap)
    # Get the excess emission
    col_sum_df["excess"] = col_sum_df[output_cols[0]] - (carbon_cap * num_gen)
    #Only take the minimum of that value and zero
    col_sum_df["Output"] = col_sum_df["excess"].clip(lower = 0)
    return(col_sum_df["Output"])

'''
def sum_tail(arr: np.ndarray, divisor: int):
    """Given a 3D vector, sum and take the average
    
    Parameters
    -----------
    arr: np.ndarray
        An 3D array of values we want to sum across the first dimension
    divisor: int
        The divisor

    Returns
    -------
    float
        The average of the elements in the depth dimension. 
    """ 
    #Currently only works for 3d aray
    return(np.sum(arr, axis = 0) * (1/divisor))
'''
    
#Function that creates cohort based on categorical varibales
def cluster_from_cat(df: pd.Series):
    """
    Given a dataframe with categorical variables corresponding to each unique generator
    create a numerical representation of the categorical column to use as the cluster

    Parameters
    -----------
    df: pd.Series
        dataframe with the unique id as the index
    cat_column: str
        string of the column name that contains the categories

    Returns
    -----------
    df: pd.Series
        series with new column called clusters. And saves it
    """
    df["Cluster"] = pd.factorize(df.iloc[:,0])[0]
    pd.DataFrame(df).to_csv("cohort.csv")
    return(df["Cluster"])

def scen_file_list(folder_path: str) -> list:
    """
    Given a folder, will extract all the csv files contained in that folder. It helps if the files
    share a common pre-fix (i.e. file_1.csv, file_2.csv, ...)

    Parameters
    -----------
    folder_path: str
        absolute path of the folder containing all the csv files

    Returns
    -----------
    list: 
        list of the absolute path of all the csvs. 
    int:
        length of the file list
    """
    #Make the folder into a path object
    folder_path = Path(folder_path)

    #Open all emission files
    #iterate over directory and store filenames in a list
    #Only keep the csv files and put file names in a list
    files_list = list(filter(lambda f: f.name.endswith('.csv'), folder_path.iterdir()))

    return([files_list, len(files_list)])

def scen_import(f_list: list, cols: list[str],suffix_char: list[int]) -> pd.DataFrame:
    """Given the list of file paths, a list of columns that we want to use in the data:
    the col identifying column of the data, the indices to be used when importing data,
    and the columns used for computation, then creates a dataframe of all the files
    
    Parameters
    -----------
    folder_path: str
        the name of the folder path containing the scenario files
    cols: list
        in each of the scenario files, what is the column name for the 
        column that contains the ID, name of index columns of the file,
        and column names to perform computation on
    suffix_char: list
        the start and end positions of the characters in the file name that index
        the different files


    Returns
    -------
    pd.DataFrame
        dataframe holding all the csv in the file list
    """
    #Read all into a csv
    em_df = (pd.read_csv(f, sep = ",", usecols = cols,
                        header = 0).assign(Scenario = f.name[slice(suffix_char[0],suffix_char[1], 1)]) 
                        for f in f_list)
    #Concat into a dataframe from a generator class
    #return(em_df)
    return(pd.concat(em_df, ignore_index=True))
