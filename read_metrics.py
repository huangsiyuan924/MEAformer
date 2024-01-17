# encoding=utf-8
import os
import sys

model_name = sys.argv[1]
dataset_name = sys.argv[2]

file_start = f"{model_name}_{dataset_name}"
if "former" in file_start:
    data_dir = "logs/LongForecasting_formers"
else:
    data_dir = "logs/LongForecasting"
file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(file_start)]
for file in file_list:
    try:
        last_line = open(file, 'r', encoding='utf-8').readlines()[-1]
    except:
        continue

    print(f"{file}----{last_line}")