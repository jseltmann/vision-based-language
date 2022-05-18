import csv
import os
from translate_table import translate_table


if __name__=="__main__":
    #data_base_path = "/home/jseltmann/data/results_with_sg_fixed_cap_binary"
    data_base_path = "/home/jseltmann/data/results_coco_val_binary_addition"
    all_models_dict = dict()
    order = []
    order_cap = []
    with open("/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv", newline='') as combf:
        not_comment = lambda line: line[0]!='#'
        reader = csv.reader(filter(not_comment, combf), delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            suite_name = row[4]
            if "ade" in suite_name:
                continue
            cap_suite_name = row[4] + "_cap"
            order.append(suite_name)
            order_cap.append(cap_suite_name)

    results_dict = dict()
    for model in ["bert", "visual-bert", "w2v"]:
        csv_path = os.path.join(data_base_path, model+".csv")
        #print(csv_path)
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for line in reader:
                if line[0] == "model":
                    continue
                suite_name = line[0].strip()
                if not suite_name in results_dict:
                    results_dict[suite_name] = dict()
                results_dict[suite_name][model] = round(float(line[1]),2)

    #order_orig = order
    with open("by_cap_table.txt", "w") as cf:
        for sn, snc in zip(order, order_cap):
            if not sn in results_dict or not snc in results_dict:
                continue
            template, strategy, other = translate_table(sn, 0)
            s = template + " & "+ strategy + " & " + other + " & "
            n = results_dict[sn]["bert"]
            c = results_dict[snc]["bert"]
            diff = round(c - n,3)
            s += str(n) + " & " + str(c) + " & " + str(diff) + " & "
            n = results_dict[sn]["visual-bert"]
            c = results_dict[snc]["visual-bert"]
            diff = round(c - n,3)
            s += str(n) + " & " + str(c) + " & " + str(diff) + " & "
            n = results_dict[sn]["w2v"]
            c = results_dict[snc]["w2v"]
            diff = round(c - n,3)
            s += str(n) + " & " + str(c) + " & " + str(diff) + "\\\\\\hline\n"
            cf.write(s)

    results_dict = dict()
    data_base_path = "/home/jseltmann/data/results_coco_val_sg"
    for model in ["bert-base", "visual-bert-base", "w2v"]:
        results_dict[model] = dict()
        csv_path = os.path.join(data_base_path, model+"_sg.csv")
        #print(csv_path)
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for line in reader:
                if line[0] == "model":
                    continue
                suite_name = line[0].strip()
                if not suite_name in results_dict:
                    results_dict[suite_name] = dict()
                results_dict[suite_name][model] = round(float(line[1]),2)

    #order_orig = order
    with open("by_cap_table_sg.txt", "w") as cf:
        for sn, snc in zip(order, order_cap):
            if not sn in results_dict or not snc in results_dict:
                continue
            template, strategy, other = translate_table(sn, 0)
            s = template + " & "+ strategy + " & " + other + " & "
            n = results_dict[sn]["bert-base"]
            c = results_dict[snc]["bert-base"]
            diff = round(c - n,3)
            s += str(n) + " & " + str(c) + " & " + str(diff) + " & "
            n = results_dict[sn]["visual-bert-base"]
            c = results_dict[snc]["visual-bert-base"]
            diff = round(c - n,3)
            s += str(n) + " & " + str(c) + " & " + str(diff) + " & "
            n = results_dict[sn]["w2v"]
            c = results_dict[snc]["w2v"]
            diff = round(c - n,3)
            s += str(n) + " & " + str(c) + " & " + str(diff) + "\\\\\\hline\n"
            cf.write(s)


