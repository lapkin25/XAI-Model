import numpy as np
import scipy

def compare_samples(sample1, sample2):
    print("Sample 1:  mean %.5f" % np.mean(sample1))
    print("Sample 2:  mean %.5f" % np.mean(sample2))
    print('Манн-Уитни p-value = %.26f' % scipy.stats.mannwhitneyu(sample1, sample2).pvalue)

def load_sample_from_file(file_name):
    arr = []
    with open(file_name, "r") as file:
        lines = file.readlines()
        for v in lines:
            number = float(v.strip())
            arr.append(number)
    #print(len(arr))
    #print(arr)
    return arr

sample1 = load_sample_from_file("sample_phenotype.txt")
sample2 = load_sample_from_file("sample_linear.txt")
compare_samples(sample1, sample2)