import json
import pickle
import os
import csv
from collections import defaultdict
import numpy as np
import math

from translate_table import translate_table
from compare_binary_sg import order_all

examples_base_path = "/home/jseltmann/data/suites_coco_val"

#examples_same_obj = "ade_tfidf_same_obj_test.json"
with open("/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv", newline='') as combf:
    not_comment = lambda line: line[0]!='#'
    reader = csv.reader(filter(not_comment, combf), delimiter=",")

    with open("sizes_one_column.csv", "w") as sf:
        #writer = csv.writer(sf, delimiter='|')
        #header = ["suite", "model", "relevant region"]
        #header += ["correct" + str(i) for i in range(5)]
        #writer.writerow(header)
        sf.write("template & strategy & other & train set & test set & template & strategy & other & train set & test set\\\\\n")
        lines = dict()
        for i, row in enumerate(reader):
            if i == 0:
                continue
            for add in ["", "_cap"]:
                suite_name = row[4] + add
                if add == "_cap" and suite_name.startswith("ade"):
                    continue
                template, strategy, other = translate_table(suite_name, 0)
                if "v-sim" in strategy:
                    continue
                train_path = row[3] + "/" + suite_name + "_train.json"
                train_path = os.path.join(examples_base_path, train_path)
                if not os.path.exists(train_path):
                    train_num = 0
                else:
                    with open(train_path) as tf:
                        d = json.load(tf)
                        train_num = len(d)

                test_path = row[3] + "/" + suite_name + "_test.json"
                test_path = os.path.join(examples_base_path, test_path)
                if not os.path.exists(test_path):
                    test_num = 0
                else:
                    with open(test_path) as tf:
                        d = json.load(tf)
                        test_num = len(d)
                lines[suite_name] = [template, strategy, other, str(train_num), str(test_num)]
        order = order_all()
        ordered_lines = []
        for sn in order:
            if not sn in lines:
                continue
            ordered_lines.append(lines[sn])
        cut = math.ceil(len(ordered_lines) / 2)
        lines1 = ordered_lines#ordered_lines[:cut]
        lines2 = []#ordered_lines[cut:]
        for i in range(len(lines1)):
            if i < len(lines2):
                l1 = lines1[i]
                l2 = lines2[i]
                s1 = " & ".join(l1)
                s2 = " & ".join(l2)
                s = s1 + " & " + s2 + "\\\\\n"
                sf.write(s)
            else:
                l1 = lines1[i]
                s1 = " & ".join(l1)
                #s = s1 +  " & & & & \\\\\n"
                s = s1 +  "  \\\\ \\hline \n"
                sf.write(s)
                #sf.write(suite_name + " & " + template + " & " + strategy + " & " + other + " & " + str(train_num) + " & " + str(test_num) + "\\\\\n")
                #sf.write(template + " & " + strategy + " & " + other + " & " + str(train_num) + " & " + str(test_num) + "\\\\\n")


