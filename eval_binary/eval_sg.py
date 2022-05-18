import csv
import sys
import json
import os
import pickle
import syntaxgym as sg
import lm_zoo


def eval_pairwise():

    for model_name in ["bert-base", "w2v", "visual-bert-base"]: 
        if model_name != "w2v":
            model = lm_zoo.get_registry()[model_name]
        results_path = os.path.join("/home/jseltmann/data/results_coco_val_sg", model_name+"_sg.csv")
        pred_base_path = "/home/jseltmann/data/results_coco_val_sg/" + model_name
        if not os.path.exists(pred_base_path):
            os.makedirs(pred_base_path)
        with open(results_path, "a+") as rf:
            writer = csv.writer(rf, delimiter="|")
            writer.writerow(["model", "acc", "prec", "recall", "tn", "fp", "fn", "tp", "pos ex.", "neg ex."])

            comb_path = "/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv"
            with open(comb_path, newline='') as gcf:
                not_comment = lambda line: line[0]!='#'
                reader = csv.reader(filter(not_comment, gcf), delimiter=",")
                for i, row in enumerate(reader):
                    for add in ["", "_cap"]:
                        suite_name = row[4] + add
                        results = dict()
                        if i == 0:
                            continue
                        pred_path = os.path.join(pred_base_path, model_name + "-"+suite_name+"_sg.pkl")
                        if os.path.exists(pred_path):
                            continue
                        with open("done.log", "a+") as df:
                            df.write(model_name + " " + suite_name + "\n")
                        if model_name == "w2v":
                            curr_model_name = model_name + "-" + suite_name
                            if not curr_model_name in lm_zoo.get_registry():
                                continue
                            model = lm_zoo.get_registry()[curr_model_name]
                        suites_dir = os.path.join("/home/jseltmann/data/suites_coco_val", row[3])
                        suite_path = os.path.join(suites_dir, suite_name + "_suite.json")
                        if not os.path.exists(suite_path):
                            continue

                        try:
                            with_surprisals = sg.compute_surprisals(model, suite_path)
                        except Exception as e:
                            with open("errors.log", "a+") as ef:
                                ef.write(model_name + " " + suite_name + "\n")
                                ef.write(str(e))
                                ef.write("\n--------\n\n")
                                continue
                        eval_results = sg.evaluate(with_surprisals)
                        results = eval_results["result"].values.tolist()

                        with open(pred_path, "wb") as pp:
                            pickle.dump((eval_results, with_surprisals), pp)

                        acc = sum(results) / len(results)
                        pad_str = " " * (20 - len(suite_name))
                        writer.writerow([suite_name+pad_str, acc, "", "", "", "", "", "", "", ""])

if __name__=="__main__":
    eval_pairwise()
