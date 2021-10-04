import os
import random
import json
import nltk
import inflect
import configparser

from generation_utils import FoilPair, get_ade_paths

random.seed(0)

def ade_thereis_generator(config_path):
    """
    Generate context based on ADE20k dataset.
    Each examples gives the scene name and an object contained in it,
    e.g. "This is a street. There is an umbrella.".

    Parameters
    ----------
    config_path : str
        Path to config file containing information about
        data to be used and the examples to be created.

    Return
    ------
    pairs : [FoilPair]
        List of FoilPairs with the foil examples not yet set.
    """

    config = configparser.ConfigParser()
    config.read(config_path)
    dataset_name = config["General"]["dataset"]
    data_path = config["Datasets"][dataset_name]
    num_examples = int(config["General"]["num_examples"])

    p = inflect.engine()
    pairs = []
    img_info_paths = get_ade_paths(data_path)

    base_img_paths = random.choices(img_info_paths, k=num_examples)
    for imgp in base_img_paths:
        annot = json.load(open(imgp))['annotation']
        scene = random.choice(annot['scene'])
        synset = nltk.corpus.wordnet.synsets(scene)[0]
        if synset.pos() == "n":
            context = "This is " + p.a(scene) + "."
        else:
            context = "This is " + p.a(scene) + " place."

        #obj = random.choice(annot['object'])
        #obj_name = obj['raw_name']
        #correct = context + p.a(obj_name) + "."

        pair = FoilPair(context, imgp)
        pairs.append(pair)

    return pairs


def vg_attribute_generator(config_path):
    """
    Generate context based on Visual Genome dataset.
    Each examples gives an object with an attribute,
    e.g. "This is a green clock.".

    Parameters
    ----------
    config_path : str
        Path to config file containing information about
        data to be used and the examples to be created.

    Return
    ------
    pairs : [FoilPair]
        List of FoilPairs with the foil examples not yet set.
    """

    config = configparser.ConfigParser()
    config.read(config_path)
    dataset_name = config["General"]["dataset"]
    vg_path = config["Datasets"][dataset_name]
    num_examples = int(config["General"]["num_examples"])

    vg_attr_path = os.path.join(vg_path, "attributes.json")
    with open(vg_attr_path) as vg_attr_f:
        attr_data = json.loads(vg_attr_f.read())

    pairs = []

    base_imgs = random.choices(attr_data, k=num_examples)
    for img in base_imgs:
        obj = random.choice(img['attributes'])
        context = ["There is ", obj["synsets"][0].split(".")[0]]
        pair = FoilPair(context, img["image_id"])
        pairs.append(pair)

    return pairs
