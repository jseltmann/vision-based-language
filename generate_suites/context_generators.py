import os
import random
import json
import inflect
import configparser
import spacy
import pickle

#import stanza
#import nltk

import generation_utils as gu

random.seed(0)

def ade_thereis_generator(pairs, config):
    """
    Generate context based on ADE20k dataset.
    Each examples gives the scene name and an object contained in it,
    e.g. "This is a street. There is an umbrella.".

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

    p = inflect.engine()
    nlp = spacy.load("en_core_web_sm")

    ade_path = config["Datasets"]["ade_path"]
    with open(os.path.join(ade_path, "index_ade20k.pkl"), "rb") as indf:
        index = pickle.load(indf)

    for pair in pairs:
        orig_dir = index["folder"][pair.orig_img]
        json_name = pair.orig_img.split(".")[0] + ".json"
        json_path = os.path.join(orig_dir, json_name)
        annot = json.load(open(json_path))['annotation']
        scene = random.choice(annot['scene'])
        doc = nlp(scene)
        if doc[0].pos_ == "NOUN":
            context = ("This is " + p.a(scene) + ".",)
        else:
            context = ("This is " + p.a(scene) + " place.",)

        pair.context = context

    return pairs


def vg_attribute_generator(pairs, config):
    """
    Generate context based on Visual Genome dataset.
    Each examples gives an object with an attribute,
    e.g. "This is a green clock.".

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

    attrs = gu.attrs_as_dict(config, keys="visgen")

    for pair in pairs:
        with_attrs = [o for o in img['attributes'] if o['attributes'] != 0]
        obj = random.choice(with_attrs)
        word = obj["synsets"][0].split(".")[0]
        context = ("There is ", word)
        pair.context = context
        pair.info["orig_object"] = obj

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
