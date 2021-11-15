import os
import random
import json
import inflect
import configparser
import spacy
import pickle
from tqdm import tqdm
import codecs
import copy

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
    data_path = "/".join(ade_path.split("/")[:-1])
    with open(os.path.join(ade_path, "index_ade20k.pkl"), "rb") as indf:
        index = pickle.load(indf)

    #for pair in tqdm(pairs):
    for pair in pairs:
        json_path = gu.get_ade_json_path(pair.orig_img, data_path, index)
        with codecs.open(json_path, "r", "ISO-8859-1") as jfile:
            annot = json.load(jfile)['annotation']
        scene = random.choice(annot['scene'])
        doc = nlp(scene)
        if doc[0].pos_ == "NOUN":
            context = ("This is " + p.a(scene) + ".",)
        else:
            context = ("This is " + p.a(scene) + " place.",)

        pair.context = context

    return pairs


def qa_base_generator(pairs, config):
    """
    Use question as context.
    """

    qa_pairs = gu.vg_as_dict(config, "question_answers", keys="visgen")

    new_pairs = []
    for pair in pairs:
        orig_id = pair.orig_img
        qas = qa_pairs[orig_id]['qas']
        qas = random.choices(qas, k=10)
        for qa in qas:
            new_pair = copy.deepcopy(pair)
            new_pair.context = qa["question"]

            r1 = qa["question"]
            new_pair.correct["regions"].append({"region_number":1, "content": r1})
            r2 = qa["answer"]
            new_pair.correct["regions"].append({"region_number":2, "content": r2})

            new_pair.region_meta = {"1": "question", "2": "answer"}
            new_pair.formula = "(*;%foiled%) > (*;%correct%)"
            new_pairs.append(new_pair)

    return new_pairs


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

    attrs = gu.vg_as_dict(config, "attributes", keys="visgen")
    new_pairs = []

    for pair in pairs:
        img = attrs[pair.orig_img]
        with_attrs = [o for o in img['attributes'] if 'attributes' in o]
        for obj in with_attrs:
            if len(obj['synsets']) == 0:
                continue
            word = random.choice(obj['synsets']).split(".")[0]
            context = ("There is ", word)
            new_pair = copy.deepcopy(pair)
            new_pair.context = context
            new_pair.info["orig_object"] = obj
            new_pairs.append(new_pair)

    return new_pairs


def relationship_obj_generator(pairs, config):
    """
    Use a relationship from Visual Genome as basis
    for the pair, inserting the foil word for the
    object of the relation.
    
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

    rels = gu.vg_as_dict(config, "relationships", keys="visgen")
    new_pairs = []
    for pair in pairs:
        orig_rels = rels[pair.orig_img]
        for rel in orig_rels["relationships"]:
            if 'name' in rel['subject']:
                subj = rel['subject']['name']
            else:
                subj = rel['subject']['names'][0]
            pred = rel['predicate']
            context = (subj + " " + pred,)
            new_pair = copy.deepcopy(pair)
            new_pair.context = context
            if 'name' in rel['object']:
                new_pair.info = {"orig_obj": rel['object']['name']}
            else:
                new_pair.info = {"orig_obj": rel['object']['names'][0]}
            new_pairs.append(new_pair)
    return new_pairs


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

    #vg2coco = gu.get_vg_image_ids(config)

    nlp = spacy.load("en_core_web_sm")
    #for pair in tqdm(pairs):
    for pair in pairs:
        #img = caption_dict[vg2coco[pair.orig_img]]
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


def vg_obj_list_generator(pairs, config):
    """
    Generate context as list of objects in the image.
    """

    objs = gu.vg_as_dict(config, "objects", keys="visgen")
    p = inflect.engine()

    new_pairs = []
    for pair in pairs:
        orig_objs = objs[pair.orig_img]["objects"]
        obj_names = [o["names"][0] for o in orig_objs]
        objs_per_example = 5
        while len(obj_names) > objs_per_example:
            new_pair = copy.deepcopy(pair)
            curr_objs = obj_names[:objs_per_example]
            obj_names = obj_names[objs_per_example:]

            context_objs = curr_objs[:(objs_per_example-1)]
            context = "There is "
            for obj in context_objs:
                context += p.a(obj) + ","
            context += "and "
            new_pair.context = context
            new_pair.info = {"corr_obj": obj_names[-1]}

            new_pairs.append(new_pair)

    return new_pairs
