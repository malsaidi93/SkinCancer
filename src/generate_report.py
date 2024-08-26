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

def ApplySoftmax(forSoftmax):
    headers_transpose = ["Filename", "RandomColorJitter", "RandomGrayScale", "RandomHorizontalFlip", "RandomRotation", "RandomVerticalFlip"]
    forSoftmax = np.array(forSoftmax).T
    softmax_data = softmax(forSoftmax, axis=1)
    tableSoftmax = []
    for idx, value in enumerate(softmax_data):
        row = [headers[idx+1]]
        row.extend(softmax_data[idx])
        tableSoftmax.append(row)
    
    print("="*10+"Softmax Probabilities Per Class"+"="*10)
    print(tabulate(tableSoftmax, headers=headers_transpose, tablefmt="pretty"))
    df = pd.DataFrame(tableSoftmax, columns=['Filename', 'RandomColorJitter', 'RandomGrayScale', 'RandomHorizontalFlip', 'RandomRotation', 'RandomVerticalFlip'])
    df.to_excel('../reports/Combined_softmax.xlsx', index=False)
    return None

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
    
    print("="*10+"F1-Scores"+"="*10)
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    
    ApplySoftmax(forSoftmax)
    