import csv
import os
from translate_table import translate_table


if __name__=="__main__":
    #data_base_path = "/home/jseltmann/data/results_with_sg_fixed_cap_binary"
    data_base_path = "/home/jseltmann/data/results_coco_val_binary_addition"
    sg_base_path = "/home/jseltmann/data/results_coco_val_sg"
    all_models_dict = dict()
    order = []
    order_cap = []
    #with open("/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv", newline='') as combf:
    #    not_comment = lambda line: line[0]!='#'
    #    reader = csv.reader(filter(not_comment, combf), delimiter=",")
    #    for i, row in enumerate(reader):
    #        if i == 0:
    #            continue
    #        suite_name = row[4]
    #        if not "ade" in suite_name:
    #            continue
    #        #cap_suite_name = row[4] + "_cap"
    #        order.append(suite_name)
    #        #order_cap.append(cap_suite_name)

    results_dict = dict()
    strategies = set()
    others = set()
    templates = set()
    for model in ["bert", "visual-bert", "w2v"]:
        csv_path = os.path.join(data_base_path, model+".csv")
        #print(csv_path)
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for line in reader:
                if line[0] == "model":
                    continue
                suite_name = line[0].strip()
                template, strategy, other = translate_table(suite_name, 0)
                if not "Tfidf" in strategy and not "Freq" in strategy:
                    continue
                strategies.add(strategy)
                others.add(other)
                templates.add(template)
                if not other in results_dict:
                    results_dict[other] = dict()
                if not strategy in results_dict[other]:
                    results_dict[other][strategy] = dict()
                results_dict[other][strategy][model] = round(float(line[1]),2)

    print(strategies)
    print(others)
    print(templates)
    9 / 0

    results_dict_sg = dict()
    for model in ["bert-base", "visual-bert-base", "w2v"]:
        csv_path = os.path.join(sg_base_path, model+"_sg.csv")
        #print(csv_path)
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file, delimiter="|")
            for line in reader:
                if line[0] == "model":
                    continue
                suite_name = line[0].strip()
                template, strategy, other = translate_table(suite_name, 0)
                if not "Tfidf" in strategy and not "Freq" in strategy:
                    continue
                strategies.add(strategy)
                others.add(other)
                if not other in results_dict_sg:
                    results_dict_sg[other] = dict()
                if not strategy in results_dict_sg[other]:
                    results_dict_sg[other][strategy] = dict()
                results_dict_sg[other][strategy][model] = round(float(line[1]),2)

    #order_strategies = ["ADE-SI", "ADE-SC", "ADE-DC", "Tfidf-Cat", "Tfidf-Sce", "Tfidf-Obj", "Freq-Cat", "Freq-Sce", "Freq-Obj"]
    order_templates = ["QA", "Rel-obj", "Obj-list", "Cap-adj", "Obj-attr"]
    #order_templates = ["Rel-obj", "Obj-list", "Cap-adj", "Obj-attr", "QA"]
    order_strategies = ["CXC-same", "CXC-sim","CXC-dis","CXC-v-dis", "VS"]
    #order_others = ["cat", "sce", "sce no-occ", "cat no-occ"]
    order_others = [""]
    with open("by_closeness_vg.txt", "w") as cf:
        for strat in order_strategies:
            for model in ["bert", "visual-bert", "w2v"]:
                if model == "visual-bert":
                    line = strat + " & VisualBERT"
                elif model == "bert":
                    line = " & BERT"
                elif model == "w2v":
                    line = " & GloVe"
                for template in order_templates:
                    if not template in results_dict:
                        line += " & "
                    elif not strat in results_dict[template]:
                        line += " & "
                    elif not "CXC-same" in results_dict[template]:
                        line += " & "
                    elif not model in results_dict[template][strat]:
                        line += " & "
                    elif not model in results_dict[template]["CXC-same"]:
                        line += " & "
                    else:
                        acc = results_dict[template][strat][model]
                        acc_same = results_dict[template]["CXC-same"][model]
                        diff = round(acc - acc_same,3)
                        line += " & " + str(diff)
                for template in order_templates:
                    if model == "bert":
                        model = "bert-base"
                    elif model == "visual-bert":
                        model = "visual-bert-base"
                    if not template in results_dict_sg:
                        line += " & "
                    elif not strat in results_dict_sg[template]:
                        line += " & "
                    elif not "CXC-same" in results_dict_sg[template]:
                        line += " & "
                    elif not model in results_dict_sg[template][strat]:
                        line += " & "
                    elif not model in results_dict_sg[template]["CXC-same"]:
                        line += " & "
                    else:
                        acc = results_dict_sg[template][strat][model]
                        acc_same = results_dict_sg[template]["CXC-same"][model]
                        diff = round(acc - acc_same,3)
                        line += " & " + str(diff)
                line += "\\\\\n"
                cf.write(line)
            cf.write("\\hline\n")
