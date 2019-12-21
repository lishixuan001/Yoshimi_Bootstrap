import numpy as np
import pandas as pd
from os.path import join
from pdb import set_trace as st
from numpy import random
from scipy import stats
import sys
import pickle



data_type = "0506"
file_path = f"./lsx_{data_type}.csv"
ages = [18, 31, 51, np.inf]
stages = ["normal_bp", "elevated", "stage_1", "stage_2", "stage_3"]
iterations = 1000

def report_progress(progress, total, lbar_prefix = '', rbar_prefix=''):
    percent = round(progress / float(total) * 100, 2)
    buf = "{0}|{1}| {2}{3}/{4} {5}%".format(lbar_prefix, ('#' * round(percent)).ljust(100, '-'),
        rbar_prefix, progress, total, percent)
    sys.stdout.write(buf)
    sys.stdout.write('\r')
    sys.stdout.flush()

def operate():
    # Load CSV
    df = pd.read_csv(file_path)

    # Start Empty Table for Track Result
    result = dict()

    for i in range(len(ages)-1):
        start_age, end_age = ages[i], ages[i+1] # [0, 19]

        print("\n=============================================")
        print(f"Age {start_age} ~ {end_age}")
        print("=============================================")

        df_age = df[df['age'] >= start_age]
        df_age = df_age[df_age['age'] < end_age]

        distances = list()
        for i in range(iterations):
            
            groups = [df_age[df_age[stage] == True] for stage in stages] # [Sample_size, 7200] * Num_Stages
            groups = [group[group.columns[2:-6]] for group in groups]
            groups = [group for group in groups if group.shape[0] > 0]
            
            # Get sample size
            N = 100
            
            # Get random indices
            random_indices = [random.choice(group.shape[0], N) for group in groups]
            
            # Get samples
            samples = [groups[j].iloc[random_indices[j], :] for j in range(len(groups))]
            
            # Get medians
            medians = [pd.DataFrame.median(sample, axis=0) for sample in samples]
            for j in range(len(medians)):
                medians[j][pd.isnull(medians[j])] = 0
            
            # Get distances
            dists = []
            for j in range(len(medians)-1):
                for k in range(j+1, len(medians)):
                    dist = max(medians[j] - medians[k])
                    dists.append(dist)
            distance = float(np.mean(dists))
            distances.append(distance)
        
            report_progress(i, iterations)

        distances = np.array(distances)
        mean, sigma = np.mean(distances), np.std(distances)
        conf_int = stats.norm.interval(0.999, loc=mean, scale=sigma)
                
        t_value = (mean - 0) / (sigma) # TODO / np.sqrt(len(distances))) # t-statistic for mean
        pval = stats.t.sf(np.abs(t_value), len(distances) - 1) * 2  # two-sided pvalue = Prob(abs(t)>tt)
        result[(start_age, end_age)] = pval
        print(f"Age [{start_age}~{end_age}]; P-Value [{pval}]")
    
    with open(f"./{data_type}_general_results.pkl", "wb") as file:
        pickle.dump(result, file)
    return print(result)

if __name__=='__main__':
    operate()
