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

    with open("surprisals_diffs.csv", "w", newline='') as sf:
        writer = csv.writer(sf, delimiter='|')
        header = ["suite", "model", "mean", "std"]
        #header += ["correct" + str(i) for i in range(5)]
        writer.writerow(header)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            results_dict = dict()
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
                for model in ["bert-base", "visual-bert-base"]:
                    pred_path = model + "/" + model + "-" + suite_name + "_sg.pkl"
                    pred_path = os.path.join(results_base_path, pred_path)
                    if not os.path.exists(pred_path):
                        continue
                    with open(pred_path, "rb") as pred_file:
                        _, suite = pickle.load(pred_file)
                    ref_regions = suite.predictions[0].referenced_regions
                    ref_region = list(ref_regions)[0][1]
                    surprisals = defaultdict(lambda: defaultdict(list))
                    diffs = []
                    for item in suite.items:
                        for condition in item['conditions']:
                            cn = condition['condition_name']
                            for region in condition['regions']:
                                rn = region['region_number']
                                if rn != 2:
                                    continue
                                surp = region['metric_value']['mean']
                                if cn == 'correct':
                                    scorr = surp
                                else:
                                    sfalse = surp
                        diff = scorr - sfalse
                        diffs.append(diff)
                    mean = round(np.mean(diffs),3)
                    std = round(np.std(diffs),3)

                    output = []
                    output.append(suite_name)
                    output.append(model)
                    output.append(mean)
                    output.append(std)
                    writer.writerow(output)
