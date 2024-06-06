# shapley_allocations

shapley_allocations is a Python library for computing exact symmetric [shapley](https://en.wikipedia.org/wiki/Shapley_value) allocations of coalitional games efficiently. It uses the matrix representation from Wang[^1] et al (2018) for the computation. The library specializes in coalitional games where the characteristic function is random and has to be simulated via Monte Carlo methods. In addition, it is able to handle chharacteristic functions that vary across the time dimension.

## Installation

Make sure Python 3.9 or later is installed.

Clone the repo with the following command:
```bash
git clone git@github.com:PrincetonUniversity/shapley_allocations.git
```

Change directory to the newly created `shapley_allocations` directory.

Install shapley_allocations with [pip](https://pip.pypa.io/en/stable/):

```bash
pip install .
```

## Usage

### Shapley Intuition
Suppose there is a game with $N=3$ coalitions, labelled $K_1, K_2, K_3$. Let their payoff function $v: 2^3 \rightarrow \mathbb{R}$ be such that 
```math
v(K) =
\begin{cases}
\begin{array}{cc}
  & 
    \begin{array}{cc}
      1 & K \in \{K_1, K_2, K_3\} \\
      2 & K \in \{K_1 \cup K_2, K_1 \cup K_3, K_2 \cup K_3\} \\
      3 & K = K_1 \cup K_2 \cup K_3
    \end{array}
\end{array}\end{cases}.
```
The characteristc value in this case expresses that the characteristc value equals the number of coalitions. Thus in this case, a fair way to distribute the characteristc value for each coalitions marginal contribution to the system characteristc value is by rewarding each participating coalition a unit of the characteristc value. It is not always the case that the marginal contibution is additive in this way, and that is where the Shapley allocations is introduced to handle such cases.

### Emission Examples
Our examples are motivated by the carbon emissions of power generators in the electric grid. We devise a method to implement a cap-n-trade carbon trading mechanism between the producers. The resulting costs is our output for the characteristic functions of the differerent coalition of power generators. 

Additionally, our emissions data is generated via Monte Carlo simulations due to randomness in the way the data is processed. For this reason, there is a distribution of emissions for some time frame. We simply take the worst case emissions and average them (i.e. [CVaR](https://en.wikipedia.org/wiki/Expected_shortfall))

### Running the full program on command line
Installing shapley adds the command shapley_allocations to your command line namespace. The simplest way to invoke this command from the terminal is:

```bash
python -m shapley_allocations.main --folder_path $folder_path --area_fname $area_fname --output_file $output_file --num_cohorts $num_cohorts --alpha $alpha --num_scen $num_scen --group_cols $group_cols --output_cols $output_cols --asset_id $asset_id --sec_asset_id $sec_asset_id --cluster_fname $cluster_fname --char_func $char_func
```

`folder_path` is the path of the folder containing all of the simulation.

`area_fname` is the name of the file containing the ids of all the players in the game.

`output_file` is the file destination for the results of the Shapley allocations.

`num_cohorts` is the number of distinct coalitons.

`alpha` is a threshold for the value at risk calculation. It is interpreted as the tail losses. Default=0.05.

`num_scen` is the the number of Monte Carlo simulation. Each simulation should be in a different file. Default=500.

`group_cols` is the index used in the scenario files. The index names are separated by a single comma. Default=`"Scenario,Hour"`.

`output_cols` is a list of column names that are used in computing the characteristic values. Passed as a string separated by a comma. Default=`"CO2 Emissions metric ton,Dispatch,Unit Cost"`.

`asset_id` is the column name of the `area_fname` that contains the players id. Default=`"GEN UID"`

`sec_asset_id` is the column name of the simulation files containing the player id. Default=`"Generator"`

`cluster_fname` if there is a premade cluster file already. If not, leave as `""`. Default=`""`.

`char_func` is the characteristic function to be used. Pass `"rate"/"carbon_excess"/"carbon_price"`.

`allowance` (OPTIONAL) is the carbone allowence for each generator. This value is used only if char_func is carbon_excess or carbon_price.", default=50

`is_absolute` (OPTIONAL) is whether the carbon allownce is relative (default) or absolute (if this is passed). This value is used only if char_func is carbon_excess or carbon_price.

`price_fname` (OPTIONAL) is a path to a .csv file containing the LMP prices. This value is used only if char_func is carbon_price. Default=`""`.

### Using the package from a Python script
You can use the package from a Python script:
```python
from shapley_allocations.shapley.shapley import calc_shapley_var
calc_shapley_var(folder_path=folder_path, output_cols=output_cols, group_cols=group_cols,
                     sec_asset_id=sec_asset_id, alpha=alpha, output=char_func,
                     cluster_fname=cluster_fname, num_cohorts=num_cohorts, area_fname=area_fname, asset_id=asset_id,
                     num_scen=num_scen, output_file=output_file, allowance=allowance,
                     is_absolute=is_absolute, price=price)
```

### Determining the worst case simulations
One way that we optimize the code, in terms of memory, is that we only focus on the worst case simulations. For instance, if there are 500 simulations, and we select the 5\% worst simulations resulting in 25 simulations.

```python
from shapley_allocations.shapley import gen_bau_scen, tail_sims

# First, we create a dataframe that reads in csv files from a folder path. The number of .csv files are the num_scen. We specify what the index of the csv files should be, and also the columns of interest if we do not want all columns
folder_path = "root/directory"
num_scen = 500
group_cols = ["Scenario","Hour"] #Does not have to be a unique index, #We use a multi-index now because we want the output specfying for each hour, the 100*alpha % worst case scenarios.

output_cols = ["CO2 Emissions metric ton", "Dispatch"]
sec_asset_id = ["Generator"]

#Returns a dataframe with the emission rate for each hour in all scenarios
files_list, num_files = scen_file_list(folder_path)
scen_em_df = scen_import(files_list,output_cols + [group_cols[1]] + [sec_asset_id], [5,9])

#Next, we filter out the highest emission rates
alpha = 0.05
#The output of this is a list of the (Scenario, Hour) pairs that show the worst case simulations
tail_scen = tail_sims(alpha, calc_col(scen_em_df, output_cols, sec_asset_id, group_cols, 
                                      output, **output_kwargs), group_cols)
#In order to locate the worst scenarios, we filter through the indices
scen_em_df.set_index(group_cols, inplace= True)

#Filter for only the tail scenarios
scen_em_df = scen_em_df.loc[tail_scen,:] #The final parameter is an arbitrary column name for

```

### Clustering
For now, any meaningful clustering is completed outside of the code. This is to keep the program flexible. Users are able to import their premade clusters into a dataframe.

There is capabilities of crude clustering: given a dataframe of $I$ unique id's, the 
'''python 
gen_cohort(num_cohorts, fname, indx)
'''
can cluster the players into $N=I/K$ even clusters.

```python
from shapley_allocations.shapley import gen_bau_scen, tail_sims

# First, we read in the cohorts file. This is a csv with two columns. The first column is the ID of the player and the second column is the corresponding cluser they are in.
cluster_fname = "root/directory_to_cohort.csv"

#Returns a dataframe of the cohorts
cohort_df = cluster_from_cat(pd.read_csv(cluster_fname,
                                                 header = 0, index_col = 0))
#Next we aggregate the cluster values accordingly
group_cluster_df = cluster_values(scen_em_df, sec_asset_id, cohort_df, group_cols,
                                  output_cols, ["sum"] * len(output_cols))
```

### Shapley Allocations
We compute the characteristic matrix and shapley allocations
```python
num_cohorts = 3
sec_asset_id = "Generator"


#Next, we compute our payoffs into what is called the characteristic matrix. 
#The next function computes the emission rate in each coalition (2^N) for the worst case scenarios
char_matrix = gen_cohort_payoff(group_cluster_df, num_cohorts, 
                                group_cols, output_cols, sec_asset_id, output, **output_kwargs)
#Compute the var of the shapley values of each simulation
result = var_shap(np.append(shap(char_matrix, num_cohorts), 
                            char_matrix[:,0].reshape(len(tail_scen),1), axis = 1),
                  tail_scen, group_cols, np.ceil(alpha*num_scen))
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## References
[^1]: Wang, Y., Cheng, D. & Liu, X. Matrix expression of Shapley values and its application to distributed resource allocation. Sci. China Inf. Sci. 62, 22201 (2019). https://doi.org/10.1007/s11432-018-9414-5
