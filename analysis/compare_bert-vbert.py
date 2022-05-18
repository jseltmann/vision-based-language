import csv
import os
from translate_table import translate_table


if __name__=="__main__":
    #data_base_path = "/home/jseltmann/data/results_with_sg_fixed_cap_binary"
    data_base_path = "/home/jseltmann/data/results_coco_val_binary_addition"
    all_models_dict = dict()
    order = []
    for model in ["bert", "visual-bert", "w2v"]:
        line_dict = dict()
        csv_path = os.path.join(data_base_path, model+".csv")
        #print(csv_path)
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
            #all_models_dict[line_name][model] = [line_name, model] + [line[1]]
            all_models_dict[line_name][model] = round(float(line[1]), 2)

    results = []
    for line_name in all_models_dict:
        bert = all_models_dict[line_name]["visual-bert"]
        w2v = all_models_dict[line_name]["w2v"]
        diff = round(w2v - bert,3)
        results.append((line_name, diff))

    results = sorted(results, key=lambda t: t[1], reverse=True)
    with open("vbert-w2v-binary.txt", "w") as f:
        for suite_name, diff in results:
            f.write(suite_name + "\t\t\t" + str(diff) + "\n")


    order_orig = order

    #data_base_path = "/home/jseltmann/data/results_with_sg_fixed_cap"
    data_base_path = "/home/jseltmann/data/results_coco_val_sg"
    sg_models_dict = dict()
    order = []
    for model in ["bert", "visual-bert", "w2v"]:
        line_dict = dict()
        if model in ["bert", "lxmert", "visual-bert"]:
            csv_path = os.path.join(data_base_path, model+"-base_sg.csv")
        else:
            csv_path = os.path.join(data_base_path, model+"_sg.csv")
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
            if not line_name in sg_models_dict:
                sg_models_dict[line_name] = dict()
            line = line_dict[line_name]
            #sg_models_dict[line_name][model] = [line_name, model] + [line[1]]
            sg_models_dict[line_name][model] = round(float(line[1]), 2)

    results = []
    for line_name in sg_models_dict:
        bert = sg_models_dict[line_name]["visual-bert"]
        w2v = sg_models_dict[line_name]["w2v"]
        diff = round(w2v - bert,3)
        results.append((line_name, diff))

    results = sorted(results, key=lambda t: t[1], reverse=True)
    with open("vbert-w2v-sg.txt", "w") as f:
        for suite_name, diff in results:
            f.write(suite_name + "\t\t\t" + str(diff) + "\n")
