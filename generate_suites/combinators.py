import os
import random
import json
import inflect
import copy
import pickle
import codecs

import generation_utils as gu

random.seed(0)

def ade_thereis_combinator(pairs, config):
    """
    Create correct and foil texts based on pairs.
    Select one object each to be correct and foil text.

    Parameters
    ----------
    pairs : [FoilPair]
        List of FoilPairs with selected foil images.

    Return
    ------
    pairs_with_texts : [FoilPair]
        Pairs with correct and foil texts.
    """

    ade_path = config["Datasets"]["ade_path"]
    data_path = "/".join(ade_path.split("/")[:-1])
    with open(os.path.join(ade_path, "index_ade20k.pkl"), "rb") as indf:
        index = pickle.load(indf)

    pairs_with_texts = []
    p = inflect.engine()
    for pair in pairs:
        json_path = gu.get_ade_json_path(pair.orig_img, data_path, index)
        with codecs.open(json_path, "r", "ISO-8859-1") as jfile:
            orig_annot = json.load(jfile)['annotation']

        orig_obj = random.choice(orig_annot['object'])
        orig_obj_name = orig_obj['raw_name']
        correct_text = "There is " + p.a(orig_obj_name) + "."
        pair.correct["regions"].append({"region_number":1, "content":pair.context[0]})
        pair.correct["regions"].append({"region_number":2, "content":correct_text})

        json_path = gu.get_ade_json_path(pair.foil_img, data_path, index)
        with codecs.open(json_path, "r", "ISO-8859-1") as jfile:
            foil_annot = json.load(jfile)['annotation']
        foil_obj = random.choice(foil_annot['object'])
        foil_obj_name = foil_obj['raw_name']
        foiled_text = "There is " + p.a(foil_obj_name) + "."
        pair.foiled["regions"].append({"region_number":1, "content":pair.context[0]})
        pair.foiled["regions"].append({"region_number":2, "content":foiled_text})

        pair.region_meta = {"1": "context", "2": "thereis"}
        pair.formula = "(2;%foiled%) > (2;%correct%)"

        pairs_with_texts.append(pair)

    return pairs_with_texts


def vg_attribute_combinator(pairs, config):
    """
    For context of type "There is an [attribute] thing".
    """

    attrs = gu.vg_as_dict(config, "attributes", keys="visgen")
    p = inflect.engine()
    new_pairs = []

    for pair in pairs:
        orig_obj = pair.info["orig_object"]
        attr = random.choice(orig_obj["attributes"])
        r = pair.context[0] + p.a(attr)
        r += " " + pair.context[1] + "."
        pair.correct['regions'].append({"region_number":1, "content": r})

        img_attrs = attrs[pair.foil_img]['attributes']
        has_attrs = lambda o: 'attributes' in o.keys() and o['attributes'] != []
        objs = [obj for obj in img_attrs if has_attrs(obj)]
        for obj in objs:
            #obj = random.choice(objs)
            attr = random.choice(obj['attributes'])
            r = pair.context[0] + p.a(attr)
            r += " " + pair.context[1] + "."
            new_pair = copy.deepcopy(pair)
            new_pair.foiled['regions'].append({"region_number":1, "content": r})

            new_pair.region_meta = {"1": "sentence"}
            new_pair.formula = "(1;%foiled%) > (1;%correct%)"
            new_pairs.append(new_pair)

    return new_pairs


def caption_adj_combinator(pairs, config):
    """
    Generate foil text by replacing an adjective
    with an attribute from a different context.
    """

    attrs = gu.attrs_as_dict(config)
    p = inflect.engine()

    full_pairs = []
    for pair in pairs:
        img_attrs = attrs[pair.foil_img]['attributes']
        has_attrs = lambda o: 'attributes' in o.keys() and o['attributes'] != []
        objs = [obj for obj in img_attrs if has_attrs(obj)]
        for obj in objs:
            new_pair = copy.deepcopy(pair)
            attr = random.choice(obj['attributes'])

            earlier, later = new_pair.context
            if 'start' in new_pair.info and new_pair.info['start'] == True:
                new_pair.foiled["regions"].append({"region_number":1, "content": ""})
                r2 = attr.capitalize().strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            elif new_pair.info['indefinite']:
                r1 = earlier + " " + p.a(attr).strip()
                new_pair.foiled["regions"].append({"region_number":1, "content": r1})
                r2 = attr.strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            else:
                r1 = earlier.strip()
                new_pair.foiled["regions"].append({"region_number":1, "content": r1})
                r2 = earlier.strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            r3 = later.strip()
            new_pair.foiled["regions"].append({"region_number":3, "content": r3})
            full_pairs.append(new_pair)

    return full_pairs


def relationship_obj_combinator(pairs, config):
    """
    Counterpart to relationship_obj_generator.
    """

    objs = gu.vg_as_dict(config, "objects", keys="visgen")

    full_pairs = []
    for pair in pairs:
        foil_objs = objs[pair.foil_img]['objects']
        for obj in foil_objs:
            new_pair = copy.deepcopy(pair)

            r1 = pair.context[0]
            new_pair.correct["regions"].append({"region_number": 1, "content": r1})
            r2 = pair.info["orig_obj"]
            new_pair.correct["regions"].append({"region_number": 2, "content": r2})

            new_pair.foiled["regions"].append({"region_number": 1, "content": r1})
            if "name" in obj:
                r2 = obj["name"]
            else:
                r2 = obj["names"][0]
            new_pair.foiled["regions"].append({"region_number": 2, "content": r2})

            new_pair.region_meta = {"1": "context", "2": "object"}
            new_pair.formula = "(*;%foiled%) > (*;%correct%)"
            full_pairs.append(new_pair)

    return full_pairs
