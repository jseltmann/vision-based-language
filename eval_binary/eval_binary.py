import csv
import sys
import json
import os
import pickle
import sklearn.metrics as metrics
sys.path.append("./w2v")
from eval_w2v import W2V_classifier
sys.path.append("./bert")
from eval_bert import BERT_classifier
sys.path.append("./lxmert")
from eval_lxmert import LXMERT_classifier


def eval_pairwise():
    results_path = "/home/jseltmann/data/results_5k_pairwise_january_shuffle_fixed"

    for model_name in ["w2v", "bert", "lxmert"]:
        results_path = os.path.join("/home/jseltmann/data/results_with_sg_fixed", model_name+".csv")
        pred_base_path = "/home/jseltmann/data/results_with_sg_fixed/" + model_name
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
                    results = dict()
                    if i == 0:
                        continue
                    #if not row[3] == "caption_pairs":
                    #    continue
                    print(model_name, row[4])
                    #if not (row[1] == "vg_obj_list_generator" or \
                    #        row[2] in ["ade_same_category_combinator", "ade_same_object_combinator", "ade_thereis_combinator"]):
                    #    continue
                    if model_name == "w2v":
                        model_path = "w2v/w2v-" + row[4] + ".pkl"
                        if not os.path.exists(model_path):
                            continue
                        model = W2V_classifier(model_path)
                    elif model_name == "bert":
                        model_path = "bert/bert-" + row[4] + ".pkl"
                        if not os.path.exists(model_path):
                            continue
                        model = BERT_classifier(model_path)
                    elif model_name == "lxmert":
                        model_path = "lxmert/lxmert-" + row[4] + ".pkl"
                        if not os.path.exists(model_path):
                            continue
                        model = LXMERT_classifier(model_path)
                    else:
                        raise("Not implemented")
                    suites_dir = os.path.join("/home/jseltmann/data/suites_with_sg_fixed", row[3])
                    suite_path = os.path.join(suites_dir, row[4] + "_test.json")

                    with open(suite_path) as testf:
                        test_examples = json.load(testf)
                    labels = []
                    predictions = []

                    for ex in test_examples:
                        labels.append(ex[1])
                        pred = model.classify(ex[0])
                        predictions.append(pred)

                    pred_path = os.path.join(pred_base_path, model_name + "-"+row[4]+".pkl")
                    with open(pred_path, "wb") as pp:
                        pickle.dump(predictions, pp)

                    acc = metrics.accuracy_score(labels, predictions)
                    prec = metrics.precision_score(labels, predictions)
                    rec = metrics.recall_score(labels, predictions)
                    tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions).ravel()

                    #with open(results_path, "a+") as rf:
                    pad_str = " " * (20 - len(row[4]))
                    writer.writerow([row[4]+pad_str, acc, prec, rec, tn, fp, fn, tp, tp+fn, tn+fp])

if __name__=="__main__":
    eval_pairwise()
