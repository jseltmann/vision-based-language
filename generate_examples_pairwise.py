import csv
import json
import random
import os

random.seed(25)

def print_examples(comb_path, out_path, num_examples=5):
    """
    Write examples of the suites listed in comb_path to out_path.
    """

    with open(comb_path, newline='') as gcf, open(out_path, 'w') as of:
        not_comment = lambda line: line[0] != '#'
        reader = csv.reader(filter(not_comment, gcf), delimiter=",")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            for suffix in ["", "_cap"]:
                #suites_dir = os.path.join("/home/jseltmann/data/suites_with_sg_fixed_cap", row[3])
                suites_dir = os.path.join("/home/jseltmann/data/suites_coco_val", row[3])
                suite_name = row[4] + suffix
                suite_path = os.path.join(suites_dir, suite_name + "_test.json")
                if not os.path.exists(suite_path):
                    continue
                with open(suite_path) as sf:
                    test_items = json.load(sf)
                train_path = os.path.join(suites_dir, suite_name + "_train.json")

                with open(train_path) as sf:
                    suite = json.load(sf)
                #items = suite['items']
                if len(suite) == 0:
                    of.write(suite_name + "no examples\n\n")
                    continue
                chosen_items = random.choices(suite, k=num_examples)
                examples = []
                for item in chosen_items:
                    if item[1] == 0:
                        corr_text = item[0][0]
                        foil_text = item[0][1]
                    else:
                        corr_text = item[0][1]
                        foil_text = item[0][0]
                    #conds = item['conditions']
                    #corr = [c for c in conds if c['condition_name']=='correct'][0]
                    #corr_text = ""
                    #for r in corr["regions"]:
                    #    corr_text += r['content']
                    #    corr_text += " "
                    #corr_text = corr_text.strip()

                    #foiled = [c for c in conds if c['condition_name']=='foiled'][0]
                    #foil_text = ""
                    #for r in foiled["regions"]:
                    #    foil_text += r['content']
                    #    foil_text += " "
                    #foil_text = foil_text.strip()

                    examples.append((corr_text, foil_text))

                of.write(suite_name + "\n")
                of.write("number of test items: " + str(len(test_items)) + "\n")
                #with open(train_path) as tf:
                #    train_data = json.load(tf)
                #    train_examples = train_data["true"] + train_data["false"]
                of.write("number of train examples: " + str(len(suite)) + "\n")
                for corr, foiled in examples:
                    of.write(corr + " - " + foiled + "\n")
                of.write("\n\n")


if __name__=="__main__":
    print_examples("generate_suites/generation_combinations_pairwise.csv",
                   "suite_examples_coco_val.txt", num_examples=10)
