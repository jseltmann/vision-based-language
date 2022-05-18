import csv
import os

#data_base_path = "/home/jseltmann/data/results_with_sg_fixed_cap_binary"
data_base_path = "/home/jseltmann/data/results_coco_val_binary_addition"

all_models_dict = dict()

order = []

for model in ["bert", "visual-bert", "w2v"]:
    line_dict = dict()
    csv_path = os.path.join(data_base_path, model+".csv")
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter="|")
        for line in reader:
            if line[0] == "model":
                continue
            line_new = []
            for i, entry in enumerate(line):
                if i > 1 and len(entry)>0:
                    entry = round(float(entry), 2)
                line_new.append(entry)
            line = line_new
            line_dict[line[0].strip()] = line
            if not line[0].strip() in order:
                order.append(line[0].strip())
    for line_name in order:
        if not line_name in all_models_dict:
            all_models_dict[line_name] = dict()
        line = line_dict[line_name]
        all_models_dict[line_name][model] = [line_name, model] + line[1:]

with open("binary_results_table.csv", "w", newline='') as tablef:
    writer = csv.writer(tablef, delimiter="|")
    for suite_name in order:
        for model in ["bert", "visual-bert", "w2v"]:
            line = all_models_dict[suite_name][model]
            writer.writerow(line)


