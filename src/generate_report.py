import os
import json
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.special import softmax

def main(root, class_names):
    combined_reports = {}
    for file in os.listdir(root):
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                report = json.load(f)
            combined_reports[file] = {each_class: report[each_class]['f1-score'] 
                                      for each_class in report if each_class in class_names}
    return combined_reports

def SoftmaxProbs(aug, softmax_values):
    headers = ["Filename", "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    table = []
    for idx, probs in enumerate(softmax_values):
        row = [aug[idx]]
        row.extend(probs)  # No need for probs[idx] since probs is already a list                                                 
        table.append(row)
    print(tabulate(table, headers=headers, tablefmt="pretty"))

if __name__ == "__main__":
    root = "../reports/"  # Specify the correct path
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    combined_reports = main(root, class_names)

    # Convert combined_reports to DataFrame for Excel output
    df_f1_scores = pd.DataFrame.from_dict(combined_reports, orient='index', columns=class_names)

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

    print("="*10+" F1-Scores "+"="*10)                                              
    print(tabulate(table, headers=headers, tablefmt="pretty"))                    

    # Calculate softmax across the columns (axis=0)
    softmax_data = softmax(forSoftmax, axis=0)                                    

    print("="*10+" Softmax Values "+"="*10)                                              
    SoftmaxProbs(aug, softmax_data)                                               
    
    # Convert softmax data to DataFrame for Excel output
    df_softmax = pd.DataFrame(softmax_data, index=aug, columns=class_names)

    # Write both DataFrames to Excel in different sheets
    with pd.ExcelWriter('../reports/combined_reports.xlsx') as writer:
        df_f1_scores.to_excel(writer, sheet_name='F1-Scores')
        df_softmax.to_excel(writer, sheet_name='Softmax Values')

