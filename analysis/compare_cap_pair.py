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
                if template != "cap-pair":
                    continue
                #if other != "":
                #    continue
                strategies.add(strategy)
                others.add(other)
                templates.add(template)
                if not strategy in results_dict:
                    results_dict[strategy] = dict()
                if not model in results_dict[strategy]:
                    results_dict[strategy][model] = dict()
                if not other in results_dict[strategy][model]:
                    results_dict[strategy][model][other] = dict()
                results_dict[strategy][model][other] = round(float(line[1]),2)

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
                if template != "cap-pair":
                    continue
                strategies.add(strategy)
                others.add(other)
                if not strategy in results_dict_sg:
                    results_dict_sg[strategy] = dict()
                if not model in results_dict_sg[strategy]:
                    results_dict_sg[strategy][model] = dict()
                if not other in results_dict_sg[strategy][model]:
                    results_dict_sg[strategy][model][other] = dict()
                results_dict_sg[strategy][model][other] = round(float(line[1]),2)

    #order_strategies = ["ADE-SI", "ADE-SC", "ADE-DC", "Tfidf-Cat", "Tfidf-Sce", "Tfidf-Obj", "Freq-Cat", "Freq-Sce", "Freq-Obj"]
    #order_templates = ["QA", "Rel-obj", "Obj-list", "Cap-adj", "Obj-attr"]
    #order_templates = ["Rel-obj", "Obj-list", "Cap-adj", "Obj-attr", "QA"]
    order_strategies = ["CXC-same", "CXC-sim","CXC-dis","CXC-v-dis"]
    #order_others = ["cat", "sce", "sce no-occ", "cat no-occ"]
    order_others = ["cxc-low", "cxc-high", "jacc-low", "jacc-high"]
    with open("by_jacc_cap_pair.txt", "w") as cf:
        for strat in order_strategies:
            line = strat
            for model in ["bert", "visual-bert", "w2v"]:
                if not strat in results_dict:
                    line += " & "
                elif not model in results_dict[strat]:
                    line += " & "
                elif not "jacc-low" in results_dict[strat][model]:
                    line += " & "
                elif not "jacc-high" in results_dict[strat][model]:
                    line += " & "
                else:
                    acc_low = results_dict[strat][model]["jacc-low"]
                    acc_high = results_dict[strat][model]["jacc-high"]
                    diff = round(acc_low - acc_high, 3)
                    line += " & " + str(diff)
            for model in ["bert-base", "visual-bert-base", "w2v"]:
                if not strat in results_dict_sg:
                    line += " & "
                elif not model in results_dict_sg[strat]:
                    line += " & "
                elif not "jacc-low" in results_dict_sg[strat][model]:
                    line += " & "
                elif not "jacc-high" in results_dict_sg[strat][model]:
                    line += " & "
                else:
                    acc_low = results_dict_sg[strat][model]["jacc-low"]
                    acc_high = results_dict_sg[strat][model]["jacc-high"]
                    diff = round(acc_low - acc_high, 3)
                    line += " & " + str(diff)
            line += "\\\\\n"
            cf.write(line)
            cf.write("\\hline\n")
