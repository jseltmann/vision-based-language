import json
import configparser
import sys
import csv
import copy
import os
import random
import logging
import traceback
import shutil
import pickle

import context_generators as cg
import foil_image_selectors as fis
import combinators as com
import generation_utils as gu

def generate_suite(pairs, save_path, suite_name, functions):
    """
    Turn pairs produced by the combinator into a syntaxgym suite.

    Parameters
    ----------
    pairs : [FoilPair]
        Duh.
    save_path : str
        Filepath to save the suite to.
    suite_name : str
        Name of suite.
    functions : (str, str, str)
        Names of (generator, selector, combinator) functions used to create suite.
    """

    suite = dict()
    suite["meta"] = dict()
    suite["meta"]["name"] = suite_name
    suite["meta"]["metric"] = "mean"
    suite["meta"]["comments"] = dict()
    suite["meta"]["comments"]["functions"] = functions
    suite["predictions"] = [{"type": "formula", "formula": pairs[0].formula}]
    suite["region_meta"] = pairs[0].region_meta
    suite["items"] = []

    for i, pair in enumerate(pairs):
        item = dict()
        item["item_number"] = i
        item["conditions"] = [pair.correct, pair.foiled]
        suite["items"].append(item)

    with open(save_path, "w") as suite_file:
        json.dump(suite, suite_file)


def split(config, pairs):
    """
    Shuffle the pairs and split off some for fine-tuning.
    """
    random.shuffle(pairs)
    train_split = float(config["General"]["trainsplit"])
    num_examples = int(config["General"]["num_examples"])
    if len(pairs) >= 2 * num_examples:
        cutoff = num_examples
    else:
        cutoff = int(train_split * len(pairs))
    test = pairs[:cutoff]
    train = pairs[cutoff:]
    train_examples = int(config["General"]["train_examples"])
    train = train[:train_examples]

    finetune_path = config["General"]["finetune_path"]
    train_examples = []
    for pair in train:
        corr_text = ""
        for region in pair.correct["regions"]:
            corr_text += region["content"]
            if corr_text != "":
                corr_text += " "
        corr_text = corr_text.strip()

        incorr_text = ""
        for region in pair.foiled["regions"]:
            incorr_text += region["content"]
            if incorr_text != "":
                incorr_text += " "
        incorr_text = incorr_text.strip()

        example = (corr_text, incorr_text)
        train_examples.append(example)
    zeros = train_examples[:int(len(train_examples)/2)]
    ones = train_examples[int(len(train_examples)/2):]
    train_examples = []
    for (corr_text, incorr_text) in zeros:
        train_examples.append(((corr_text, incorr_text), 0))
    for (corr_text, incorr_text) in ones:
        train_examples.append(((incorr_text, corr_text),1))

    random.shuffle(train_examples)

    with open(finetune_path, "w") as ff:
        json.dump(train_examples, ff)
    with open(finetune_path + "_raw", "wb") as rf:
        pickle.dump(train, rf)

    test_path = config["General"]["test_path"]
    test_examples = []
    for pair in test:
        corr_text = ""
        for region in pair.correct["regions"]:
            corr_text += region["content"]
            if corr_text != "":
                corr_text += " "
        corr_text = corr_text.strip()

        incorr_text = ""
        for region in pair.foiled["regions"]:
            incorr_text += region["content"]
            if incorr_text != "":
                incorr_text += " "
        incorr_text = incorr_text.strip()

        example = (corr_text, incorr_text)
        test_examples.append(example)

    zeros = test_examples[:int(len(test_examples)/2)]
    ones = test_examples[int(len(test_examples)/2):]
    test_examples = []
    for (corr_text, incorr_text) in zeros:
        test_examples.append(((corr_text, incorr_text), 0))
    for (corr_text, incorr_text) in ones:
        test_examples.append(((incorr_text, corr_text),1))

    random.shuffle(test_examples)

    with open(test_path, "w") as ff:
        json.dump(test_examples, ff)
    with open(test_path + "_raw", "wb") as rf:
        pickle.dump(test, rf)

    return test, train


if __name__ == "__main__":

    logging.basicConfig(filename='pairwise.log', level=logging.DEBUG)

    if len(sys.argv) != 2:
        print("Using default config file.")
        config_path = "pairwise.config"
    else:
        config_path = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(config_path)

    with open("generation_combinations_pairwise.csv", newline='') as gcf:
        not_comment = lambda line: line[0]!='#'
        reader = csv.reader(filter(not_comment, gcf), delimiter=",")

        for i, row in enumerate(reader):
            try:
                cfg_copy = copy.deepcopy(config)
                if i == 0:
                    continue
                print(row[4] + ", ", end='', flush=True)
                logging.info(row[4])

                if row[1] != '':
                    cfg_copy["Functions"]["generator"] = row[1]
                generator_str = cfg_copy["Functions"]["generator"] # read name of generator function from config file
                generator = getattr(cg, generator_str)

                if row[0] != '':
                    cfg_copy["Functions"]["selector"] = row[0]
                selector_str = cfg_copy["Functions"]["selector"]
                selector = getattr(fis, selector_str)

                if row[2] != '':
                    cfg_copy["Functions"]["combinator"] = row[2]
                combinator_str = cfg_copy["Functions"]["combinator"]
                combinator = getattr(com, combinator_str)

                suite_dir = cfg_copy["General"]["suites_dir"] + row[3]
                cfg_copy["General"]["test_path"] = suite_dir + "/" + row[4] + "_test.json"
                cfg_copy["General"]["suite_path"] = suite_dir + "/" + row[4] + "_suite.json"
                cfg_copy["General"]["finetune_path"] = suite_dir + "/" + row[4] + "_train.json"
                cfg_copy["General"]["cfg_path"] = suite_dir + "/" + row[4] + ".cfg"

                if os.path.exists(cfg_copy["General"]["suite_path"]):
                    print()
                    continue

                cfg_copy["General"]["suite_name"] = row[4]

                if row[5] != '':
                    cfg_copy["Datasets"]["cxc_subset"] = row[5]

                if row[6] != '':
                    cfg_copy["General"]["cutoff"] = row[6]
                if row[7] != '':
                    cfg_copy["General"]["similar"] = row[7]
                if row[8] != '':
                    cfg_copy["General"]["trainsplit"] = row[8]
                if row[9] != '':
                    cfg_copy["General"]["num_examples"] = row[9]
                if row[10] != '':
                    cfg_copy["General"]["conditions"] = row[10].replace("|", "\n")
                if row[11] != '':
                    for other in row[11].split("|"):
                        cfg_copy["Other"][other] = "True"

                pairs = selector(cfg_copy)
                print("selected" + ", ", end='', flush=True)
                with_context = generator(pairs, cfg_copy)
                print("context" + ", ", end='', flush=True)
                combined = combinator(with_context, cfg_copy)
                print("combined" + ", ", end='', flush=True)

                if not os.path.exists(suite_dir):
                    os.makedirs(suite_dir)

                test, train = split(cfg_copy, combined)
                cfg_copy["General"]["train_examples"] = str(len(train))
                cfg_copy["General"]["test_examples"] = str(len(test))

                save_path = cfg_copy["General"]["suite_path"]
                suite_name = cfg_copy["General"]["suite_name"]

                if len(test) == 0:
                    print()
                    continue
                generate_suite(test, save_path, suite_name, 
                        (generator_str, selector_str, combinator_str))

                cfg_path = os.path.join(suite_dir, suite_name+".cfg")
                with open(cfg_path, "w") as configfile:
                    cfg_copy.write(configfile)

                if "add_cap_context" in cfg_copy["Other"]:
                    combined = gu.add_caption_context(combined, cfg_copy)
                    if not os.path.exists(suite_dir):
                        os.makedirs(suite_dir)

                    cfg_copy["General"]["test_path"] = suite_dir + "/" + row[4] + "_cap_test.json"
                    cfg_copy["General"]["suite_path"] = suite_dir + "/" + row[4] + "_cap_suite.json"
                    cfg_copy["General"]["finetune_path"] = suite_dir + "/" + row[4] + "_cap_train.json"
                    test, train = split(cfg_copy, combined)
                    cfg_copy["General"]["train_examples"] = str(len(train))
                    cfg_copy["General"]["test_examples"] = str(len(test))

                    save_path = cfg_copy["General"]["suite_path"]
                    suite_name = cfg_copy["General"]["suite_name"]

                    if len(test) == 0:
                        print()
                        continue
                    generate_suite(test, save_path, suite_name, 
                            (generator_str, selector_str, combinator_str))
                    print("extra context" + ", ", end='', flush=True)
                print()
            except Exception as e:
                print(e)
                with open("errors.log", "a") as ef:
                    ef.write(row[4] + "\n")
                    ef.write(str(e))
                    ef.write("\n\n")
                raise(e)
