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

def caption_adj_generator(pairs, config):
    """
    Generate context based on MSCoco dataset.
    Use a caption as context and replace an adjective with an attribute.
    
    Parameters
    ----------
    pairs : [FoilPair]
        List of pairs produced by selector function,
        which contain orig and foil image ids.
    config : Configparser
        Containing configuration.

    Return
    ------
    pairs : [FoilPair]
        List of FoilPairs with the foil examples not yet set.
    """

    coco_path = config["Datasets"]["mscoco_path"]
    with open(os.path.join(coco_path, "captions_val2014.json")) as cf:
        caption_data = json.loads(cf.read())['annotations']
        caption_dict = dict()
        for img in caption_data:
            caption_dict[img['image_id']] = img

    nlp = spacy.load("en_core_web_sm")
    for pair in pairs:
        img = caption_dict[pair.orig_img]
        doc = nlp(img['caption'])
        adj_positions = [i for i, word in enumerate(doc) if word.pos_=='ADJ']
        adj_ = [word for word in doc if word.pos_=='ADJ']

        adj_pos = random.choice(adj_positions)
        earlier = doc[:adj_pos]
        if len(earlier) == 0:
            info =  {'indefinite' : False, 'start' : True}
        elif  'Ind' in earlier[-1].morph.get('Definite'):
            # remove indefinite article,
            # since we don't know what sound the inserted word starts with
            earlier = earlier[:-1]
            info = {'indefinite': True}
        else:
            info = {'indefinite': False}
        later = doc[adj_pos+1:]
        
        context = (earlier.text, later.text)
        pair.context = context
        pair.info = info

        r1 = earlier.text
        pair.correct["regions"].append({"region_number":1, "content": r1})
        r2 = doc[adj_pos].text
        pair.correct["regions"].append({"region_number":2, "content": r2})
        r3 = later.text
        pair.correct["regions"].append({"region_number":3, "content": r3})

        pair.region_meta = {"1": "earlier", "2": "adj", "3": "later"}
        pair.formula = "(2;%foiled%) > (2;%correct%)"

    return pairs
