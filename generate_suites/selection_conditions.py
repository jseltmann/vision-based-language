import spacy
import json
import os

import generation_utils as gu


def caption_with_adj(pair, config, info=None):
    """
    Return true if the caption for the first image
    contains an adjective.

    Parameters
    ----------
    pair : (str,str)
        Ids of selected image pair.
    config : Configparser
        Configparser from selector function.
    info : dict
        Possible further information.
    """

    (i1id, i2id) = pair

    coco_dict = info['coco_captions']
    caption = coco_dict[i1id]
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(caption)
    adjs = [word for word in doc if word.pos_=='ADJ']
    if len(adjs) > 0:
        return True
    else:
        return False


def img_with_attr(pair, config, info=None):
    """
    Return True if at least one object in
    the foil image is annotated with an attribute.
    """

    (i1id, i2id) = pair

    vg_path = config["Datasets"]["vg_path"]
    mscoco_path = config["Datasets"]["mscoco_path"]

    attrs_as_dict = info['attrs']
    vg2coco = info['vg2coco']

    if not i2id in attrs_as_dict:
        return False
    else:
        img = attrs_as_dict[i2id]

    has_attr = False
    for obj in img['attributes']:
        if not 'attributes' in obj:
            continue
        if len(obj['attributes']) > 0:
            has_attr = True
            break
    return has_attr



def both_img_with_attr(pair, config, info=None):
    """
    Return True if both images are annotated with attributes.
    """

    (i1id, i2id) = pair

    vg_path = config["Datasets"]["vg_path"]
    mscoco_path = config["Datasets"]["mscoco_path"]

    attrs_as_dict = info['attrs']
    vg2coco = info['vg2coco']

    if not i2id in attrs_as_dict:
        return False
    else:
        img = attrs_as_dict[i2id]

    has_attr = False
    for obj in img['attributes']:
        if not 'attributes' in obj:
            continue
        if len(obj['attributes']) > 0:
            has_attr = True
            break

    if not has_attr:
        return False

    if not i1id in attrs_as_dict:
        return False
    else:
        img = attrs_as_dict[i1id]

    has_attr = False
    for obj in img['attributes']:
        if not 'attributes' in obj:
            continue
        if len(obj['attributes']) > 0:
            has_attr = True
            break

    return has_attr


def img_with_rel(pair, config, info=None):
    """
    Return True if the first image is annotated
    with at leat one relation between objects.
    """
    rels = info["rels"]
    i1id = pair[0]
    if not i1id in rels:
        return False
    if len(rels[i1id]["relationships"]) < 1:
        return False
    return True

def with_obj_2(pair, config, info=None):
    """
    Return True if the second image is annotated
    with at least one object.
    """
    objs = info["objs"]
    i2id = pair[1]
    if not i2id in objs:
        return False
    if len(objs[i2id]["objects"]) < 1:
        return False
    return True


def pair_in_vg(pair, config, info=None):
    """
    Return True iff both images with coco ids
    are also contained in VisualGenome.
    """
    c1id = pair[0]
    c2id = pair[1]

    coco2vg = info["coco2vg"]
    if c1id in coco2vg and c2id in coco2vg:
        return True
    else:
        return False

def have_questions(pair, config, info=None):
    """
    Return True iff both images are annotated with questions.
    """
    qas = info["qas"]

    c1id = pair[0]
    if qas[c1id]['qas'] == []:
        return False
    c2id = pair[1]
    if qas[c2id]['qas'] == []:
        return False
    return True


def have_objects(pair, config, info=None):
    """
    Return True iff both images are annotated with objects.
    """
    objs = info["objs"]

    c1id = pair[0]
    if objs[c1id]["objects"] == []:
        return False
    c2id = pair[1]
    if objs[c2id]["objects"] == []:
        return False
    return True
