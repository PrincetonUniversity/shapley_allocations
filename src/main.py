import argparse

from shapley.shapley import gen_cohorts, shap_alloc

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, help="Path to the input file.")
    parser.add_argument("--area_fname", type=str, help="Pass to the area filename csv file.")
    parser.add_argument("--output_file", type=str, help="Path to the desired output file.")
    parser.add_argument("--alpha", type=float, help="1-quantile.", default=0.05)
    parser.add_argument("--num_scen", type=int, help="Number of scenarios.", default=500)
    parser.add_argument("--num_cohorts", type=int, help="Number of cohorts.", default=3)
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

    scen_df = gen_cohorts(num_scen, num_cohorts, folder_path, area_fname)
    result = shap_alloc(num_scen, alpha, num_cohorts, scen_df)
    result.to_csv(output_file)