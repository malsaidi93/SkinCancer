from tabulate import tabulate
import os
import json

root = '../reports/'
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def main():
    combined_reports = {}
    for file in os.listdir(root):
        combined_reports[file] = {}
        with open(root+file, 'r') as f:
            report = json.load(f)
            
        for each_class in report.keys():
            if each_class in class_names:
                combined_reports[file][each_class] = report[each_class]['f1-score']

            
    return combined_reports
            
            


if __name__ == "__main__":
    combined_reports = main()
    headers = ["Filename", "akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    table = []
    for filename, values in combined_reports.items():
        row = [filename]
        row.extend(values.values())
        table.append(row)
        
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    