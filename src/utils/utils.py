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
    return  generator.split("_")[1]

def rec_coef(n: int) -> np.ndarray:
    """Given the number of players in the game, identify the vector that holds all the counting
    coeficients in the shapley value computation.
    The computation is executed with dynamic programming (more efficient than recursion).
    
    Parameters
    -----------
    n: int
        the number of players in the coop game

    Returns
    -------
    matrix
        a matrix of dimention 2^n x 1. It represents eta_k in page 5
    """
    curr_res = np.array([0])
    for curr_n in range(1, n+1):
        curr_res = np.append(curr_res + np.ones(2**(curr_n-1), dtype=int), curr_res)
    return curr_res

#Define the matrix T_i
def shap_helper(i: int,n: int) -> np.ndarray:
    """Given the player i allocation who we want to compute and the number of players in the game, 
    identify the matrix that holds the different basis vectors needed to compute the marginal
    cooperative payoffs
    
    Parameters
    -----------
    i: int
        the player i that we want to compute the allocation for
    n: int
        the number of players in the coop game

    Returns
    -------
    matrix
        a matrix of dimention 2^n x 2^(n-1). It represents T_i or equation 6
    """
    #Computed with diagonal blocks
    if i >= 1:    
        doubly_identity = np.append(np.identity(2**(n-i), dtype = int), 
                                -1 * np.identity(2**(n-i), dtype = int), axis = 0)
        dub_id_list = [doubly_identity] * (2**(i-1))
        return(block_diag(*dub_id_list))

#Define the shapley allocation for agent i
def shap_alloc_helper(i: int, n: int) -> np.ndarray:
    """Given a player i whose allocation we want to compute, and the number of players in the game, 
    compute the shapley allocation for that player
    
    Parameters
    -----------
    i: int (between 1 and n)
        the player i that we want to compute the allocation for 
    n: int
        the number of players in the coop game

    Returns
    -------
    matrix
        a 2^n x 1 matrix that will be multiplied with the  structure vector of the characteristic
        function (C) This is the summed term in equation 5 divided by n!
    """
    #Call of the helper function
    rec_val = rec_coef(n-1)
    T = shap_helper(i,n)
    #Since we are summing we create a zero vector as our inititialization
    value = np.zeros(2 ** n, dtype = int)
    #Sum accross all coalitions not containing i
    for j in range(2 ** (n-1)):
        value += (math.factorial(rec_val[j]) * (math.factorial(n-1-rec_val[j])) * T[:,j])
    return(value/ math.factorial(n))

#Create a list that gives the ordering for the charateristic function
#Turns out it is just the binary numbers in reverse order
def char_order(n: int) -> list:
    """Given the number of players n in the coop game, give the correct order of the payoff
    functions  in the structure vector C_v of the payoff function. Needs to be in the correct order
    or the matrix multiplication wont work. It turns out it is just binary numbers in reverse order
    padded with 0s. Also key to note the 00...0 means the perfect world scenario and 11...1 is BAU
    
    Parameters
    -----------
    n: int
        the number of players in the coop game

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
    order = [None] * (2 ** n)
    for i in range(2 **n):
        order[i] = format((2 **n) - i -1, '0' + str(n) + 'b')
    return(order)

#Create a function that converts the binary numbers to boolean list
#in opposite ditection because the 0s are the ones I idealize
def to_bool_list(x):
    list = []
    #Takes a string binary
    for i in x:
        list.append(bool(int(i)))
    return(list)

def ceildiv(a, b):
    return -(a // -b)

def divide_chunks(length, n):       
    # looping till length l 
    for i in range(0, len(length), n):  
        yield length[i:i + n] 
  