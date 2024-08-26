from tabulate import tabulate
import os
import json
import pandas as pd
root = '../reports/'
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

import numpy as np
from scipy.special import softmax


def main():
    combined_reports = {}
    for file in os.listdir(root):
        if file.endswith('.txt'):
            combined_reports[file] = {}
            with open(root+file, 'r') as f:
                report = json.load(f)
            
            for each_class in report.keys():
                if each_class in class_names:
                    combined_reports[file][each_class] = report[each_class]['f1-score']

    return combined_reports


if __name__ == "__main__":
    combined_reports = main()
    pd.DataFrame(combined_reports).to_excel('../reports/combined_reports.xlsx')
    headers = ["Filename", "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    table = []
    forSoftmax = []
    aug = []
    for filename, values in combined_reports.items():
        row = [filename]
        row.extend(values.values())
        table.append(row)
        aug.append(filename)
        forSoftmax.append(list(values.values()))

    print(tabulate(table, headers=headers, tablefmt="pretty"))
    
    # Tabulate & Apply Softmax to F1-Scores
    softmax_data = softmax(np.array(forSoftmax), axis=1)
    
    tableSoftmax = []
    for idx in range(0, len(forSoftmax)):
        row = [aug[idx]]
        row.extend(softmax_data[idx])
        tableSoftmax.append(row)

    print(tabulate(tableSoftmax, headers=headers, tablefmt="pretty"))
