import json
import pickle
import os
from collections import defaultdict
import csv

#examples_base_path = "/home/jseltmann/data/suites_5k_pairwise_january_shuffle_fixed/"
examples_base_path = "/home/jseltmann/data/suites_coco_val/"
#results_base_path = "/home/jseltmann/data/results_5k_pairwise_january_shuffle_fixed/"
results_base_path = "/home/jseltmann/data/results_coco_val_sg/"

#examples_same_obj = "ade_tfidf_same_obj_test.json"
with open("/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv", newline='') as combf:
    not_comment = lambda line: line[0]!='#'
    reader = csv.reader(filter(not_comment, combf), delimiter=",")

    with open("as_category_sg.txt", "w") as cf:
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if not row[3].startswith("ade"):
                continue
            if "tfidf" in row[4] or "freq" in row[4]:# or row[4].endswith("_scene"):
                continue
            suite_name = row[4]
            suite_path = row[3] + "/" + suite_name + "_test.json"
            suite_path = os.path.join(examples_base_path, suite_path)
            if not os.path.exists(suite_path):
                continue
            if "tisce" in suite_name:
                continue
            cat_totals = defaultdict(int)
            with open(suite_path) as sf:
                examples_with_labels = json.load(sf)
                labels = [l for (e,l) in examples_with_labels]
                examples = [e for (e,l) in examples_with_labels]
                total = len(examples)
                cat_preds = dict()
                for i, (example, label) in enumerate(examples_with_labels):
                    corr_text = example[label]
                    this_is = corr_text.split(".")[0]
                    cat = " ".join(this_is.split()[3:])
                    cat_preds[i] = (cat, labels[i])
                    cat_totals[cat] += 1
            #print(suite_name)
            #for model in ["bert", "lxmert", "w2v"]:
            cats = dict()
            for model in ["bert-base", "visual-bert-base", "w2v"]:
                cat_corr = defaultdict(int)
                pred_path = model + "/" + model + "-" + suite_name + "_sg.pkl"
                pred_path = os.path.join(results_base_path, pred_path)
                if not os.path.exists(pred_path):
                    continue
                with open(pred_path, "rb") as pred_file:
                    preds = pickle.load(pred_file)
                    preds = preds[0]['result'].to_list()
                all_corr = 0
                for i, pred in enumerate(preds):
                    cat, label = cat_preds[i]
                    if label == pred:
                        cat_corr[cat] += 1
                        all_corr += 1

                for cat in sorted(cat_corr):
                    acc = str(round(cat_corr[cat] / cat_totals[cat], 2))
                    if not cat in cats:
                        cats[cat] = dict()
                    if not "counter" in cats[cat]:
                        cats[cat]["counter"] = cat_totals[cat]
                    cats[cat][model] = (acc, cat_totals[cat])
            for i, cat in enumerate(cats):
                if i == 0:
                    cf.write(suite_name)
                cf.write(" & " + cat + " & ")
                cf.write(str(cats[cat]["counter"]))
                for model in ["bert-base", "visual-bert-base", "w2v"]:
                    if model in cats[cat]:
                        s = str(cats[cat][model][0])
                    else:
                        s = "no result"
                    cf.write(" & " + s)
                cf.write("\\\\")
                if i == len(cats)-1:
                    cf.write("\\hline")
                cf.write("\n")

                    #cf.write(suite_name + " " + cat + " " + model + " " + acc + " " + str(cat_totals[cat]) + "\n")
                    #print(cat, acc, cat_totals[cat])
