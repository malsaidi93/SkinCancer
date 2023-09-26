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

def get_augmentation_name()

def generate_report(path):
    over_all_report = {}
    files_list = os.listdir(path)    
    for each_file in files_list:
        with open(os.path.join('./reports', each_file), 'r') as f:
            p = parse_tojson(f.read())
            p = json.loads(p)
            # print(type(p))
        # print(p.keys())
        for class_keys in p.keys():
            # print(class_keys.keys())
            for score_keys in p[class_keys].keys():
                if f'Class_{score_keys}' == class_keys:
                    print(p[class_keys][score_keys]['f1-score'])
                    # over_all_report[each_key] = p[each_key][each_key_2]
if __name__ == '__main__':
    path = './reports/'
    generate_report(path)