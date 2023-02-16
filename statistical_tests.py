"""
Creates a .csv file "statistical_results.csv" containing results of statistical tests on .csv files in specified directory.
Tests include the Pearson, Spearman, and Kendall correlation tests.
Both the correlation statistic and P-value for each test is calculated and saved.

"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import csv

directory = ("Matches/")

def StatisticalTest(fileroot):
    """
    Performs Pearson, Spearman and Kendall statistical tests on file. 
    Column name for x variable can be "psi" or "Pweighted".
    Column name for y variable is "logBHMass".
    Returns Scipy object for each test:
    pearsonr, spearmanr, kendalltau
    """
    # Load the CSV file into a DataFrame
    dataframe = pd.read_csv(fileroot)

    # Getting relevant data from the dataframe
    x_name = 'psi'             # EFIGI pitch angle column
    x_name_alt = 'Pweighted'    # Diaz pitch angle column
    y_name = 'logBHMass'

    # Attempts to use either EFIGI or Diaz pitch angle column (psi or Pweighted)
    try:
        x_array = np.array(dataframe[x_name])
    except:
        x_array = np.array(dataframe[x_name_alt])
    y_array = np.array(dataframe[y_name])

    pearson_result=stats.pearsonr(x_array,y_array)
    spearman_result=stats.spearmanr(x_array,y_array)
    kendall_result=stats.kendalltau(x_array,y_array)

    return pearson_result, spearman_result, kendall_result

# iterate through .csv files in a directory, and perform the function Statistical test on them.
# create .csv file for results
header = ['filename', 'pearson', 'P_pearson', 'spearman', 'P_spearman', 'kendall', 'P_kendall']

with open("statistical_results.csv","w",newline='') as results_file:
    writer = csv.writer(results_file)
    writer.writerow(header)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            fileroot=(os.path.join(directory, filename))
            results = (StatisticalTest(fileroot))
            row = [filename,results[0][0],results[0][1],results[1][0],results[1][1],results[2][0],results[2][1]]
            writer.writerow(row)
        else:
            continue 

