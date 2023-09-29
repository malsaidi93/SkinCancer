import pandas as pd
import numpy as np
import os
import re
import json

def parse_tojson(data):
    # Use regular expressions to add double quotes around keys, including numeric keys
    data = re.sub(r'\'([A-Za-z_][A-Za-z0-9_]*)\'\s*:', r'"\1":', data)
    data = re.sub(r'([0-9]+):', r'"\1":', data)

    # Replace single quotes with double quotes
    data = data.replace("'", '"')
    
    return data

def get_augmentation_name(path):
    return path.split('_')[1]

def generate_report(path):
    over_all_report = {}
    files_list = os.listdir(path)    
    for each_file in files_list:
        aug = get_augmentation_name(each_file)
        with open(os.path.join('./reports', each_file), 'r') as f:
            p = parse_tojson(f.read())
            p = json.loads(p)
            # print(type(p))
        # print(p.keys())
        
        # this fragment of code is to get the f1-score for only the augmented class
        for class_keys in p.keys():
            # print(class_keys.keys())
            for score_keys in p[class_keys].keys():
                if f'Class_{score_keys}' == class_keys:
                    print(f'{aug} Class_{score_keys} {p[class_keys][score_keys]["f1-score"]}')
                    # over_all_report[each_key] = p[each_key][each_key_2]
        
        # this fragment of code is to get the highest f1-score among all the classes 
        max_f1_scores = {}
        for class_name, metrics in p.items():
            max_f1_score = -1  # Initialize with a very low value
            max_f1_score_key = None  # Initialize with None
            for key, metric in metrics.items():
                if key != 'accuracy':
                    # print(f'f1: {metric}')
                    f1_score = metric['f1-score']
                    if 'avg' not in key:
                        pass
                        # print(f"Aug: {aug} Augmented Class: {class_name} Class: {key} f1: {f1_score}")
                    if f1_score > max_f1_score:
                        max_f1_score = f1_score
                        max_f1_score_key = key
            max_f1_scores[class_name] = {'max_f1_score': max_f1_score, 'max_f1_score_key': max_f1_score_key}

        # print(max_f1_scores)
if __name__ == '__main__':
    path = './reports/'
    generate_report(path)