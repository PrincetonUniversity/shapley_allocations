import math

import numpy as np
from scipy.linalg import block_diag


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
    matrix
        a matrix of dimention 2^N x 1. It represents eta_k in page 5
    """
    curr_res = np.array([0])
    for curr_n in range(1, N+1):
        curr_res = np.append(curr_res + np.ones(2**(curr_n-1), dtype=int), curr_res)
    return curr_res

def shap_Phi(num_cohorts: int):
    #First define the capital phi matrix
    l_vector = rec_coef(num_cohorts-1).tolist()
    eta =  np.zeros(2**(num_cohorts-1))
    for j in range(2 ** (num_cohorts-1)):
        eta[j] = (math.factorial(l_vector[j]) * 
                   (math.factorial((num_cohorts-1)-l_vector[j])))

        Phi = np.zeros(( 2**num_cohorts, num_cohorts))
    for m in range(1, num_cohorts+1):
        eta_m = list(divide_chunks2(eta, m))
        for j in range(0,m+1, 2):
            Phi[j*((2**num_cohorts)//(2**(m))):(j+1)*((2**num_cohorts)//(2**(m))),
                (m-1)] = eta_m[j % m]
            Phi[(j+1)*((2**num_cohorts)//(2**(m))):(j+2)*((2**num_cohorts)//(2**(m))),
                (m-1)] = np.multiply(eta_m[j % m],-1)
    return(np.multiply(Phi, 1/math.factorial(num_cohorts)))   

#Define the matrix T_i
def shap_helper(i: int, N: int) -> np.ndarray:
    """Given the cluster n allocation who we want to compute and the number of players in the game, 
    identify the matrix that holds the different basis vectors needed to compute the marginal
    cooperative payoffs
    
    Parameters
    -----------
    i: int
        the player i that we want to compute the allocation for
    N: int
        the number of clusters in the coop game

    Returns
    -------
    matrix
        a matrix of dimention 2^N x 2^(N-1). It represents T_i or equation 6
    """
    #Computed with diagonal blocks
    if i >= 1:    
        doubly_identity = np.append(np.identity(2**(N-i), dtype = int), 
                                -1 * np.identity(2**(N-i), dtype = int), axis = 0)
        dub_id_list = [doubly_identity] * (2**(i-1))
        return(block_diag(*dub_id_list))

#Define the shapley allocation for agent i
def shap_alloc_helper(n: int, N: int) -> np.ndarray:
    """Given a cluster's shap value to compute, and the number of clusters in the game, 
    compute the shapley allocation for that cluster
    
    Parameters
    -----------
    n: int (between 1 and N)
        the player n that we want to compute the allocation for 
    N: int
        the number of clusters in the coop game

    Returns
    -------
    matrix
        a 2^N x 1 matrix that will be multiplied with the  structure vector of the characteristic
        function (C) This is the summed term in equation 5 divided by n!
    """
    #Call of the helper function
    rec_val = rec_coef(N-1)
    T = shap_helper(n,N)
    #Since we are summing we create a zero vector as our inititialization
    value = np.zeros(2 ** N, dtype = int)
    #Sum accross all coalitions not containing i
    for j in range(2 ** (N-1)):
        value += (math.factorial(rec_val[j]) * (math.factorial(N-1-rec_val[j])) * T[:,j])
    return(value/ math.factorial(N))

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

def ceildiv(a: int, b: int):
    """Given two integers, where a is the dividend, b is the divisor, obtain the smallest
     integer quotient c s.t. c*b > a.
    
    Parameters
    -----------
    a: int
        the divisor
    b: int
        the divisor

    Returns
    -------
    int
        the cieling of the quotient. 
    """
    return -(a // -b)

def divide_chunks(item_list: list, m: int): 
    """Given a list to split into m equal pieces, return the slices of the list
    
    Parameters
    -----------
    item_list: list
        list that you want to split
    m: int
        the number of equal pieces to split the length

    Returns
    -------
    list
        list of list containing the chunks of the list. 
    """      
    # looping till length l 
    for i in range(0, len(item_list), m):  
        yield item_list[i:i + m]

def divide_chunks2(item_list: list, m: int): 
    """Given a list to split into m equal pieces, return the slices of the list
    
    Parameters
    -----------
    item_list: list
        list that you want to split
    m: int
        the number of equal pieces to split the length

    Returns
    -------
    list
        list of list containing the chunks of the list. 
    """      
    # looping till length l 
    for i in range(0, len(item_list), (len(item_list)//m)):  
        yield item_list[i:i + len(item_list)//m]

#def sub_block(eta: np.ndarray, size: int,  m: int):
def output(df, output_cols):
     """Given a dataframe with only the columns that should be modified
    
    Parameters
    -----------
    df: dataFrame
        A dataframe containing the columns that should be evaluated
    output_cols: list
        list of strings containing the column names

    Returns
    -------
    float
        elementwise division of the two vectors. 
    """ 
     col_sum_df = df.sum()
     col_sum_df["Output"] = col_sum_df[output_cols[0]] / col_sum_df[output_cols[1]]
     return(col_sum_df["Output"])

#print(shap_Phi(2))
#print(char_order(2))