import json
import pickle
import os
import csv
from collections import defaultdict
import numpy as np
import math

#examples_base_path = "/home/jseltmann/data/suites_with_sg_fixed_cap"
examples_base_path = "/home/jseltmann/data/suites_coco_val"
results_base_path = "/home/jseltmann/data/results_coco_val_sg"

with open("/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv", newline='') as combf:
    not_comment = lambda line: line[0]!='#'
    reader = csv.reader(filter(not_comment, combf), delimiter=",")

    with open("surprisals_no_ties.csv", "w", newline='') as sf:
        writer = csv.writer(sf, delimiter='|')
        header = ["suite", "model", "acc without ties", "percentage of not ties"]
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
                #for model in ["bert-base", "lxmert-base", "w2v", "visual-bert-base"]:
                for model in ["bert-base", "w2v", "visual-bert-base"]:
                #for model in ["w2v"]:
                    pred_path = model + "/" + model + "-" + suite_name + "_sg.pkl"
                    pred_path = os.path.join(results_base_path, pred_path)
                    if not os.path.exists(pred_path):
                        continue
                    with open(pred_path, "rb") as pred_file:
                        _, suite = pickle.load(pred_file)
                    ref_regions = suite.predictions[0].referenced_regions
                    ref_region = list(ref_regions)[0][1]
                    surprisals = defaultdict(lambda: defaultdict(list))
                    results = []
                    for item in suite.items:
                        for condition in item['conditions']:
                            cn = condition['condition_name']
                            for region in condition['regions']:
                                rn = region['region_number']
                                if rn == 2:
                                    surp = region['metric_value']['mean']
                            if cn == "correct":
                                scorr = surp
                            else:
                                sfoil = surp
                            #surprisals[rn][cn].append(surp)
                            #surprisals[cn][rn].append(surp)
                        if sfoil > scorr:
                            results.append(True)
                        elif sfoil < scorr:
                            results.append(False)
                        else:
                            continue

                    output = []
                    output.append(model)
                    output.append(suite_name)
                    num_corr = len([e for e in results if e])
                    if len(results) > 0:
                        output.append(round(num_corr / len(results), 2))
                    else:
                        output.append(None)
                    output.append(round(len(results) / len(suite.items), 2))
                    writer.writerow(output)
