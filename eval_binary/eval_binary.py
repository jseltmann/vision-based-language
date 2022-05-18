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
from eval_bert_log_reg import BERT_log_reg_classifier
sys.path.append("./lxmert")
from eval_lxmert import LXMERT_classifier
sys.path.append("./visual-bert")
from eval_visual_bert import Visual_BERT_classifier


def eval_pairwise():

    for model_name in ["w2v", "bert", "visual-bert"]:#, "lxmert"]:
        results_path = os.path.join("/home/jseltmann/data/results_coco_val_binary_addition", model_name+".csv")
        pred_base_path = "/home/jseltmann/data/results_coco_val_binary_addition/" + model_name
        if os.path.exists(results_path):
            with open(results_path) as rf:
                lines = rf.readlines()
                names = set([line.split("|")[0].strip() for line in lines])
        else:
            names = []
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
                        pred_path = os.path.join(pred_base_path, model_name + "-"+suite_name+".pkl")
                        if os.path.exists(pred_path):
                            continue
                        print(model_name, suite_name)
                        if model_name == "w2v":
                            model_path = "w2v/log_reg_addition/w2v-" + suite_name + ".pkl"
                            if not os.path.exists(model_path):
                                continue
                            model = W2V_classifier(model_path)
                        elif model_name == "bert-log-reg":
                            model_path = "bert/log_reg_addition/bert-" + suite_name + ".pkl"
                            if not os.path.exists(model_path):
                                continue
                            model = BERT_log_reg_classifier(model_path)
                        elif model_name == "lxmert":
                            model_path = "lxmert/log_reg/lxmert-" + suite_name + ".pkl"
                            if not os.path.exists(model_path):
                                continue
                            model = LXMERT_classifier(model_path)
                        elif model_name == "visual-bert":
                            model_path = "visual-bert/log_reg_addition/vbert-" + suite_name + ".pkl"
                            if not os.path.exists(model_path):
                                continue
                            model = Visual_BERT_classifier(model_path)
                        else:
                            raise("Not implemented")
                        suites_dir = os.path.join("/home/jseltmann/data/suites_coco_val", row[3])
                        suite_path = os.path.join(suites_dir, suite_name + "_test.json")

                        with open(suite_path) as testf:
                            test_examples = json.load(testf)
                        labels = []
                        predictions = []

                        for ex in test_examples:
                            labels.append(ex[1])
                            pred = model.classify(ex[0])
                            predictions.append(pred)

                        with open(pred_path, "wb") as pp:
                            pickle.dump(predictions, pp)

                        acc = metrics.accuracy_score(labels, predictions)
                        prec = metrics.precision_score(labels, predictions)
                        rec = metrics.recall_score(labels, predictions)
                        try:
                            tn, fp, fn, tp = metrics.confusion_matrix(labels, predictions).ravel()
                            positives = tp + fn
                            negatives = tn + fp
                        except Exception as e:
                            tn, fp, fn, tp = None, None, None, None
                            positives = None
                            negatives = None

                        pad_str = " " * (20 - len(suite_name))
                        writer.writerow([suite_name+pad_str, acc, prec, rec, tn, fp, fn, tp, positives, negatives])

if __name__=="__main__":
    eval_pairwise()
