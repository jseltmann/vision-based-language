import json
import pickle
import os
import csv
from collections import defaultdict
import numpy as np
import math

from translate_table import translate_table

examples_base_path = "/home/jseltmann/data/suites_coco_val"
results_base_path = "/home/jseltmann/data/results_coco_val_sg"

#examples_same_obj = "ade_tfidf_same_obj_test.json"
with open("/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv", newline='') as combf:
    not_comment = lambda line: line[0]!='#'
    reader = csv.reader(filter(not_comment, combf), delimiter=",")

    with open("surprisals.csv", "w", newline='') as sf:
        writer = csv.writer(sf, delimiter='|')
        header = ["suite", "model", "relevant region"]
        for i in range(1,5):
            header.append("correct " + str(i) + " mean")
            header.append("correct " + str(i) + " std")
        for i in range(1,5):
            header.append("foiled " + str(i) + " mean")
            header.append("foiled " + str(i) + " std")
        #header += ["correct" + str(i) for i in range(5)]
        writer.writerow(header)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            for add in ["", "_cap"]:
                suite_name = row[4] + add
                suite_path = row[3] + "/" + suite_name + "_test.json"
                suite_path = os.path.join(examples_base_path, suite_path)
                if not os.path.exists(suite_path):
                    continue
                #if suite_name != "ade_diff_cat":
                #    continue
                #with open(suite_path) as sf:
                #    examples_with_labels = json.load(sf)
                #    labels = [l for (e,l) in examples_with_labels]
                #    examples = [e for (e,l) in examples_with_labels]
                for model in ["bert-base", "visual-bert-base", "w2v"]:
                    pred_path = model + "/" + model + "-" + suite_name + "_sg.pkl"
                    pred_path = os.path.join(results_base_path, pred_path)
                    if not os.path.exists(pred_path):
                        continue
                    with open(pred_path, "rb") as pred_file:
                        _, suite = pickle.load(pred_file)
                    ref_regions = suite.predictions[0].referenced_regions
                    ref_region = list(ref_regions)[0][1]
                    surprisals = defaultdict(lambda: defaultdict(list))
                    for item in suite.items:
                        for condition in item['conditions']:
                            cn = condition['condition_name']
                            for region in condition['regions']:
                                rn = region['region_number']
                                surp = region['metric_value']['mean']
                                #surprisals[rn][cn].append(surp)
                                surprisals[cn][rn].append(surp)

                    output = []
                    output.append(suite_name)
                    output.append(model)
                    output.append(ref_region)
                    #for rn in sorted(surprisals):
                    #    print(rn)
                    #    i = 0
                    #    for cn in sorted(surprisals[rn]):
                    #        mean_surp = round(np.mean(surprisals[rn][cn]), 2)
                    #        output.append(mean_surp)
                    #        std_surp = round(np.std(surprisals[rn][cn]), 2)
                    #        output.append(std_surp)
                    #        i += 1
                    #    for j in range(i,6):
                    #        output += ["", ""]
                    for cn in sorted(surprisals):
                        i = 0
                        if not 0 in surprisals[cn]:
                            output += ["", ""]
                            i += 1
                        for rn in sorted(surprisals[cn]):
                            surps = [s for s in surprisals[cn][rn] if not math.isnan(s)]
                            #mean_surp = round(np.mean(surprisals[rn][cn]), 2)
                            mean_surp = round(np.mean(surps), 2)
                            output.append(mean_surp)
                            #std_surp = round(np.std(surprisals[rn][cn]), 2)
                            std_surp = round(np.std(surps), 2)
                            output.append(std_surp)
                            i += 1
                        for j in range(i,4):
                            output += ["", ""]
                    writer.writerow(output)
