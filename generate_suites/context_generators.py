import os
import random
import json
import inflect
import configparser
import spacy

#import stanza
#import nltk

import generation_utils as gu

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
    img_info_paths = gu.get_ade_paths(data_path)

    nlp = spacy.load("en_core_web_sm")

    base_img_paths = random.choices(img_info_paths, k=num_examples)
    for imgp in base_img_paths:
        annot = json.load(open(imgp))['annotation']
        scene = random.choice(annot['scene'])
        doc = nlp(scene)
        if doc[0].pos_ == "NOUN":
            context = ("This is " + p.a(scene) + ".",)
        else:
            context = ("This is " + p.a(scene) + " place.",)

        pair = gu.FoilPair(context, imgp)
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

    coco_path = config["Datasets"]["mscoco_path"]

    vg2coco = gu.get_vg_image_ids(vg_path, coco_path)
    attr_data = [ad for ad in attr_data if ad['image_id'] in vg2coco]

    pairs = []

    base_imgs = random.choices(attr_data, k=num_examples)
    for img in base_imgs:
        obj = random.choice(img['attributes'])
        context = ("There is ", obj["synsets"][0].split(".")[0])
        coco_id = vg2coco[imag['image_id']]
        pair = gu.FoilPair(context, coco_id)
        pairs.append(pair)

    return pairs

def caption_adj_generator(config_path):
    """
    Generate context based on MSCoco dataset.
    Use a caption as context and replace an adjective with an attribute.
    
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

    coco_path = config["Datasets"]["mscoco_path"]
    with open(os.path.join(coco_path, "captions_val2014.json")) as cf:
        caption_data = json.loads(cf.read())['annotations']

    nlp = spacy.load("en_core_web_sm")
    pairs = []
    used_imgs = set()
    while len(pairs) < num_examples:
        img = random.choice(caption_data)
        if img['image_id'] in used_imgs:
            continue

        doc = nlp(img['caption'])

        adj_positions = [i for i, word in enumerate(doc) if word.pos_='ADJ']
        if adj_positions == []:
            continue

        adj_pos = random.choice(adj_positions)
        earlier = doc[:adj_pos]
        if 'Ind'  earlier[-1].morph.get('Definite'):
            # remove indefinite article,
            # since we don't know what sound the inserted word starts with
            earlier = earlier[:-1]
            info = {'indefinite': True}
        else:
            info = {'indefinite': False}
        later = doc[adj_pos+1:]
        
        context = (earlier.text, later.text)
        pair = gu.FoilPair(context, img['image_id'], info=info)
        used_imgs.add(img['image_id'])

    return pairs
