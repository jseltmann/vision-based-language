import os
import random
import configparser
from tqdm import tqdm
import json
import pickle

import generation_utils as gu
import selection_conditions as sc


random.seed(0)


def ade_different_category_selector(config):
    """
    Select a pair of images from ADE which are
    in a different outer category. E.g. home_or_hotel vs. transportation.

    Parameters
    ----------
    config : Configparser
        Read from config file

    Return
    ------
    pairs : [Pairs]
        FoilPairs with selected images.
    """

    ade_path = config["Datasets"]["ade_path"]
    with open(os.path.join(ade_path, "index_ade20k.pkl"), "rb") as indf:
        index = pickle.load(indf)

    num_examples = 2 * int(config["General"]["num_examples"])

    train_path = os.path.join(ade_path, "images/ADE/training/")
    categories = set(os.listdir(train_path))

    pairs = []
    fns = list(enumerate(index['filename']))
    selected_inds = set()
    while len(selected_inds) < num_examples:
        i, fn = random.choice(fns)
        if i in selected_inds:
            continue
        if not "train" in fn:
            continue
        selected_inds.add(i)
        path = index["folder"][i]
        category = path.split("/")[-2]
        found = False
        while not found:
            new_cat = random.choice([c for c in categories if c != category])
            cat_path = os.path.join(train_path, new_cat)
            scenes = os.listdir(cat_path)
            scene = random.choice(scenes)
            scene_path = os.path.join(cat_path, scene)
            foil_fns = [fn for fn in os.listdir(scene_path) if fn.endswith("jpg")]
            foil_fns = [fn for fn in foil_fns if "train" in fn]
            if foil_fns == []:
                continue
            foil_fn = random.choice(foil_fns)
            found = True
        pair = gu.FoilPair(fn, foil_fn)
        pairs.append(pair)

    return pairs


def ade_different_scene_selector(config):
    """
    Select a pair of images from ADE which are
    in a different scene type, but the same outer category.
    E.g., "classroom" and "library__indoor" under "cultural".

    Parameters
    ----------
    config : Configparser
        Read from config file

    Return
    ------
    pairs : [Pairs]
        FoilPairs with selected images.
    """

    ade_path = config["Datasets"]["ade_path"]
    with open(os.path.join(ade_path, "index_ade20k.pkl"), "rb") as indf:
        index = pickle.load(indf)

    num_examples = int(config["General"]["num_examples"])
    train_fns = [fn for fn in index['filename'] if "train" in fn]
    selected_fns = random.choices(list(enumerate(train_fns)), k=num_examples)

    train_path = os.path.join(ade_path, "images/ADE/training/")

    pairs = []
    for i, fn in selected_fns:
        path = index["folder"][i]
        category = path.split("/")[-2]
        scene = path.split("/")[-1]
        cat_path = os.path.join(train_path, category)
        scenes = [d for d in os.listdir(cat_path) if d!=scene]
        foil_scene = random.choice(scenes)
        foil_scene_path = os.path.join(cat_path, scene)
        fn_criteria = lambda fn: fn.endswith("jpg") and "train" in fn
        foil_fns = [fn for fn in os.listdir(foil_scene_path) if fn_criteria(fn)]
        foil_fn = random.choice(foil_fns)
        pair = gu.FoilPair(fn, foil_fn)
        pairs.append(pair)
    return pairs


def cxc_similar_selector(config):
    """
    Select a pair of images to be similar to 
    each other according to the cxc dataset.

    Parameters
    ----------
    config : Configparser
        Read from config file

    Values read from config file
    ----------------------------
    cxc_path : str
        Path to CxC dataset.
    num_examples : int
        Number of example pairs to choose.
    cutoff : float
        Similarity value below or above which to select.
    similar : bool
        Wether to select similar or dissimilar image pairs.
    cxc_subset : str
        Filename of specific part of CxC to use.
    conditions : [function]
        Functions returning bools that apply further conditions
        to the pairs to be selected.

    Return
    ------
    pairs : [Pairs]
        FoilPairs with selected images.
    """

    cxc_path = config["Datasets"]["cxc_path"]
    cutoff = float(config["General"]["cutoff"])
    similar = bool(config["General"]["similar"])
    cxc_subset = config["Datasets"]["cxc_subset"]
    conditions = config["General"]["conditions"].split("\n")
    if conditions[0] != '':
        cond_fns = [getattr(sc, cond) for cond in conditions]
    else:
        cond_fns = []

    similarities = gu.read_cxc(cxc_path, cxc_subset)

    if similar:
        idpairs = [idpair for idpair in similarities if similarities[idpair] > cutoff]
    else:
        idpairs = [idpair for idpair in similarities if similarities[idpair] < cutoff]

    pairs = []
    chosen = set()

    vg2coco = gu.get_vg_image_ids(config)
    coco2vg = gu.get_vg_image_ids(config, reverse=True)
    coco_dict = gu.coco_as_dict(config)
    attrs_as_dict = gu.vg_as_dict(config, "attributes", keys="coco")
    rels_dict = gu.vg_as_dict(config, "relationships", keys="coco")
    objs_dict = gu.vg_as_dict(config, "objects", keys="coco")
    qas = gu.vg_as_dict(config, "question_answers", keys="coco")

    info = {'attrs': attrs_as_dict,
            'vg2coco': vg2coco,
            'coco2vg': coco2vg,
            'coco_captions': coco_dict,
            'rels': rels_dict,
            'objs': objs_dict,
            'qas': qas}

    #for idpair in tqdm(idpairs):
    for idpair in idpairs:
        sort_out = False
        for cond_fn in cond_fns:
            if cond_fn(idpair, config, info=info) == False:
                sort_out = True
                break

        if sort_out == True:
            continue

        i1id, i2id = idpair
        if config["Functions"]["generator"] != "caption_adj_generator":
            # using VG ids is usually more useful
            i1id = coco2vg[i1id]
            i2id = coco2vg[i2id]
        else:
            # but for the captions we can use coco images
            # not contained in VisualGenome
            # the foil image still needs to be in VG, though
            i2id = coco2vg[i2id]
        pair = gu.FoilPair(i1id, i2id)
        pairs.append(pair)

    return pairs
