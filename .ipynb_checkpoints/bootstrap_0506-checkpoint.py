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
    results = dict()

    for i in range(len(ages)-1):
        start_age, end_age = ages[i], ages[i+1] # [0, 19]

        print("\n=============================================")
        print(f"Age {start_age} ~ {end_age}")
        print("=============================================")

        df_age = df[df['age'] >= start_age]
        df_age = df_age[df_age['age'] < end_age]

        result = dict()

        for j in range(len(stages)-1):
            for k in range(j+1, len(stages)):

                print("\n---------------------------------------------")
                print(f"Sets {stages[j]} & {stages[k]}")
                print("---------------------------------------------")

                # Get name of the paired sets
                cls_1 = stages[j]
                cls_2 = stages[k]

                # Get set data
                set1 = df_age[df_age[cls_1] == True]
                set2 = df_age[df_age[cls_2] == True]
                
                set1 = set1[set1.columns[2:-6]]
                set2 = set1[set2.columns[2:-6]]
                
                # Get sample size
                N = 100

                # Iterations
                distances = list()

                for i in range(iterations):
                    # Two random indices, one for each set, sample size N
                    random_index_1 = random.choice(set1.shape[0], N)
                    random_index_2 = random.choice(set2.shape[0], N)

                    # Get the two samples, each [N, 7200]
                    sample_1 = set1.iloc[random_index_1, :]
                    sample_2 = set2.iloc[random_index_2, :]

                    # Get column-wise median
                    median_1 = pd.DataFrame.median(sample_1, axis=0)
                    median_1[pd.isnull(median_1)] = 0
                    median_2 = pd.DataFrame.median(sample_2, axis=0)
                    median_2[pd.isnull(median_2)] = 0

                    distance = max(median_1 - median_2)
                    distances.append(distance)
                    report_progress(i, iterations)

                distances = np.array(distances)
                mean, sigma = np.mean(distances), np.std(distances)
                conf_int = stats.norm.interval(0.95, loc=mean, scale=sigma) # TODO / np.sqrt(len(distances)))

                t_value = (mean - 0) / (sigma) # TODO / np.sqrt(len(distances))) # t-statistic for mean
                pval = stats.t.sf(np.abs(t_value), len(distances) - 1) * 2  # two-sided pvalue = Prob(abs(t)>tt)
                result[(j, k)] = pval

                print(f"Result: [{result[(j, k)]}]")

        results[(start_age, end_age)] = result

    with open(f"./{data_type}_results.pkl", "wb") as file:
        pickle.dump(results, file)
    return results

if __name__=='__main__':
    operate()
