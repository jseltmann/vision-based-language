import json
import pickle
import os
import csv

examples_base_path = "/home/jseltmann/data/suites_5k_pairwise_january_shuffle_fixed/"
results_base_path = "/home/jseltmann/data/results_5k_pairwise_january_shuffle_fixed/"

#examples_same_obj = "ade_tfidf_same_obj_test.json"
with open("/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv", newline='') as combf:
    not_comment = lambda line: line[0]!='#'
    reader = csv.reader(filter(not_comment, combf), delimiter=",")

    for i, row in enumerate(reader):
        if i == 0:
            continue
        for add in ["", "_cap"]:
            suite_name = row[4] + add
            if not "rel_obj" in suite_name:
                continue
            suite_path = row[3] + "/" + suite_name + "_test.json"
            suite_path = os.path.join(examples_base_path, suite_path)
            with open(suite_path) as sf:
                examples_with_labels = json.load(sf)
                labels = [l for (e,l) in examples_with_labels]
                examples = [e for (e,l) in examples_with_labels]
            sg_path = row[3] + "/" + suite_name + "_suite.json"
            with open(
            for model in ["bert", "lxmert", "w2v"]:
                pred_path = model + "/" + model + "-" + suite_name + ".pkl"
                pred_path = os.path.join(results_base_path, pred_path)
                with open(pred_path, "rb") as pred_file:
                    preds = pickle.load(pred_file)


