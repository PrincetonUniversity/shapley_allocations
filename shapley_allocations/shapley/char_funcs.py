"""
This file includes all possible charactersitc functions to be used by calc_col and gen_cohort_payoff
to compute the characterstic value of each coalition.
"""

import pandas as pd


def output_rate(df: pd.DataFrame, output_cols: list[str], *args) -> pd.Series:
    """Given a dataframe grouped along some indexes, and the columns names to
      use in computing the carbon intensity (or division in general), 
      create a dataseries with the new column
    
    Parameters
    -----------
    df: pd.DataFrame
        A dataframe already grouped by that contains the columns that should be evaluated
    output_cols: list
        list of strings containing the column names
    *args: Parameters
        Needed to make this more modular


    Returns
    -------
    pd.Series
        elementwise division of the two vectors. It interprets as the heat rate or
        carbon intensity rate
    """ 
    col_sum_df = df.sum()
    col_sum_df["Output"] = col_sum_df[output_cols[0]] / col_sum_df[output_cols[1]]
    return(col_sum_df["Output"])

def output_carbon_excess(df: pd.DataFrame, output_cols: list[str], player_id_col:str,
                         allowance: float, is_absolute: bool, is_cluster: bool, *args) -> pd.Series:
    """Given a dataframe grouped along some axis, and the columns needed to
    compute the excess over the carbon allowance, return a data series with
    the corresponding amounts 
    
    Parameters
    -----------
    df: pd.DataFrame
        A dataframe containing the columns that should be evaluated
    output_cols: list
        list of strings containing the column names
    player_id_col: str

    allowance: float
        The maximum amount of carbon allowed
    is_absolute: bool
        A boolean that is true if the allowance is for an absolute carbon level
        and False is it is a carbon intensity
    is_cluster: bool

    Returns
    -------
    pd.Series
        elementwise subtraction and take the maximum between that and 0. 
    """ 
    group_operation = "sum"
    cluster_operation = "count"
    #If we are using the carbon intensity allowance then compute intensity and take
    #the average
    if not is_absolute:
        df = output_rate(df, output_cols[0:2])
        df = df.groupby(df.index)
        group_operation = "mean"
    
    if is_cluster:
        cluster_operation = "sum" 
    #First obtain the sum in each index level
    #df[player_id_col] = df[player_id_col].transform(lambda x: x.astype('bool'))
    group_df = df.agg({output_cols[0]: group_operation, player_id_col: cluster_operation}).\
                                            set_axis([group_operation, "count"], axis = 1)
    group_df["allowance"] = group_df["count"].mul(allowance)
    group_df["excess"] = group_df[group_operation] - group_df["allowance"]
    group_df["Output"] = group_df["excess"].clip(lower = 0)

    return(group_df["Output"])

def output_carbon_price(df: pd.DataFrame, output_cols: list[str], player_id_col: str,
                         allowance: float, is_absolute: bool, is_cluster: bool,
                         price_df: pd.Series, *args) -> pd.Series:
    """Given a dataframe grouped along some axis, and the columns needed to
    compute the excess over the carbon allowance, return a data series with
    the corresponding amounts 
    
    Parameters
    -----------
    df: pd.DataFrame
        A dataframe containing the columns that should be evaluated
    output_cols: list
        list of strings containing the column names
    player_id_col: str

    allowance: float
        The maximum amount of carbon allowed
    is_absolute: bool
        A boolean that is true if the allowance is for an absolute carbon level
        and False is it is a carbon intensity

    Returns
    -------
    pd.Series
        elementwise subtraction and take the maximum between that and 0. 
    """ 
    #First compute the cost per unit of production
    avg_cost_df = output_rate(df, output_cols[slice(None,0,-1)])

    #Merge with the price data
    avg_cost_df = pd.merge(left = avg_cost_df, right = price_df, how = "left", 
                          left_on= avg_cost_df.index.names[-1], right_index=True)
    
    avg_cost_df["profit"] = avg_cost_df[price_df.name] - avg_cost_df["Output"] 

    #avg_cost_df = avg_cost_df.groupby(avg_cost_df.index)
    #Obtain the carbon excess
    carb_excess_df = output_carbon_excess(df, output_cols, player_id_col, 
                                          allowance, is_absolute, is_cluster)
    #Obtain the carbon intensity
    carb_intensity_df = output_rate(df, output_cols)
    #Merge the two
    #carb_excess_df = pd.merge(left = carb_excess_df, right = carb_intensity_df, 
     #                         how = "left", left_index= True, right_index=True)
    
    #Co=ompute the remaining output
    #carb_excess_df["ratio"] = carb_excess_df / carb_intensity_df

    #Merge with avg cost
    #avg_cost_df = pd.merge(left = avg_cost_df, right = carb_excess_df, how = "left", 
     #                     left_on= True, right_index=True)

    #Compute the new profit
    #avg_cost_df["carb_cost"] = avg_cost_df["profit"] * avg_cost_df["ratio"] 
    avg_cost_df["carb_cost"] = avg_cost_df["profit"] * (carb_excess_df / carb_intensity_df)
    #df = df.groupby(df.index)

    return(avg_cost_df["carb_cost"].rename("Output"))