import csv
import sys
import json
import os
import sklearn.metrics as metrics
sys.path.append("./w2v")
from eval_w2v import W2V_classifier

def eval_pairwise():
    results_path = "/home/jseltmann/data/results_5k_pairwise"

    model_name = "w2v"
    results_path = os.path.join("/home/jseltmann/data/results_5k_pairwise", model_name+".txt")
    with open(results_path, "w") as rf:
        writer = csv.writer(rf, delimiter="|")
        writer.writerow(["model", "acc", "prec", "recall", "tn", "fp", "fn", "tp", "pos ex.", "neg ex."])

        comb_path = "/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv"
        with open(comb_path, newline='') as gcf:
            not_comment = lambda line: line[0]!='#'
            reader = csv.reader(filter(not_comment, gcf), delimiter=",")
            for i, row in enumerate(reader):
                results = dict()
                if i == 0:
                    continue
                print(row[4])
                if model_name == "w2v":
                    model_path = "w2v/w2v-" + row[4] + ".pkl"
                    model = W2V_classifier(model_path)
                else:
                    raise("Not implemented")
                suites_dir = os.path.join("/home/jseltmann/data/suites_5k_pairwise", row[3])
                suite_path = os.path.join(suites_dir, row[4] + "_test.json")
                
                with open(suite_path) as testf:
                    test_examples = json.load(testf)
                labels = []
                predictions = []

                for ex in test_examples:
                    labels.append(ex[1])
                    pred = model.classify(ex[0])
                    predictions.append(pred)

                acc = metrics.accuracy_score(labels, predictions)
                prec = metrics.precision_score(labels, predictions)
                rec = metrics.recall_score(labels, predictions)
                tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions).ravel()
                
                with open(results_path, "a+") as rf:
                    pad_str = " " * (20 - len(row[4]))
                    writer.writerow([row[4]+pad_str, acc, prec, rec, tn, fp, fn, tp, tp+fn, tn-fp])

if __name__=="__main__":
    eval_pairwise()
