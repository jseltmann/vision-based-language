import os
import random
import configparser
from tqdm import tqdm
import json
import pickle
import sys
import numpy as np

import generation_utils as gu
import selection_conditions as sc

sys.path.append("../explore_spaces")
from get_ade_most_common import get_ade_tfidf_raw_names, get_ade_frequencies_raw_names, get_ade_scenes_tfidf_raw_names, get_ade_scenes_frequencies_raw_names, get_ade_tfidf_obj_part

random.seed(0)


def ade_tfidf_same_category_selector(config):
    """
    Select two objects from a category,
    one with a high tfidf or frequency and one with a low one.
    """
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_frequencies_raw_names(obj_first=False)
    else:
        tfidfs = get_ade_tfidf_raw_names(print_tfidfs=False, obj_first=False)
    pairs = []

    for cat in tfidfs:
        if cat == "unclassified":
            continue
        objs = tfidfs[cat].items()
        objs = sorted(objs, key=lambda t: t[1])
        non_zero = [o for o in objs if o[1] != 0]
        avg = np.mean([o[1] for o in non_zero])
        above = [o[0] for o in non_zero if o[1] > avg]
        below = [o[0] for o in non_zero if o[1] < avg]
        high_tfidf = above
        low_tfidf = below
        for high in high_tfidf:
            for low in low_tfidf:
                info = {"category": cat, "orig_obj": high, "foil_obj": low}
                pair = gu.FoilPair(None, None, info=info)
                pairs.append(pair)
    return pairs


def ade_tfidf_same_scene_paper_selector(config):
    """
    Select two objects from a scene type,
    one with a high tfidf or frequency and one with a low one.
    """
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_scenes_frequencies_raw_names(obj_first=False)
    else:
        tfidfs = get_ade_scenes_tfidf_raw_names(obj_first=False, filter_animacy=True, filter_plurals=True)
    pairs = []

    use_cats = []
    if "home_or_hotel" in config["Other"]:
        use_cats.append("home_or_hotel")
    if "shopping_and_dining" in config["Other"]:
        use_cats.append("shopping_and_dining")
    if "transportation" in config["Other"]:
        use_cats.append("transportation")

    for cat in tfidfs:
        if not cat in use_cats:
            continue
        for sce in tfidfs[cat]:
            if sce.startswith("outliers"):
                continue
            objs = tfidfs[cat][sce].items()
            objs = sorted(objs, key=lambda t: t[1])
            non_zero = [o for o in objs if o[1] != 0]
            #avg = np.mean([o[1] for o in non_zero])
            #above = [o[0] for o in non_zero if o[1] > avg]
            #below = [o[0] for o in non_zero if o[1] < avg]
            #high_tfidf = above[-35:] #35 chosen because that gives a good number of examples
            high_tfidf = [o[0] for o in non_zero[-3:]]
            low_tfidf = [o[0] for o in non_zero[:1]]
            for high in high_tfidf:
                for low in low_tfidf:
                    if "use_ade_cat" in config["Other"]:
                        info = {"category": cat, "outer_cat": cat, "orig_obj": high, "foil_obj": low, "type": "scene-object"}
                    else:
                        info = {"category": sce, "outer_cat": cat, "orig_obj": high, "foil_obj": low, "type": "scene-object"}
                    pair = gu.FoilPair(None, None, info=info)
                    pairs.append(pair)
    return pairs


def ade_tfidf_object_part_paper_selector(config):
    """
    Select two object parts, one with a high and one with a low tfidf.
    """
    if "ade_frequency" in config["Other"]:
        #tfidfs = get_ade_scenes_frequencies_raw_names(obj_first=False)
        raise("Not implemented")
    else:
        tfidfs, obj_freq = get_ade_tfidf_obj_part(filter_animacy=True, filter_plurals=True)
    pairs = []

    obj_freq_filtered = dict()
    for obj in obj_freq:
        if obj in tfidfs and len(tfidfs[obj]) > 3:
            obj_freq_filtered[obj] = obj_freq[obj]

    obj_freq = sorted(list(obj_freq_filtered.items()), reverse=True, key=lambda t:t[1])
    most_common = [t[0] for t in obj_freq[:50]]

    for obj in most_common:
        parts = tfidfs[obj].items()
        parts = sorted(parts, key=lambda t: t[1])
        non_zero = [o for o in parts if o[1] != 0]
        high_tfidf = [o[0] for o in non_zero[-3:]]
        low_tfidf = [o[0] for o in non_zero[:1]]
        for high in high_tfidf:
            for low in low_tfidf:
                info = {"object": obj, "orig_part": high, "foil_part": low, "type": "object-part"}
                pair = gu.FoilPair(None, None, info=info)
                pairs.append(pair)
    return pairs


def ade_tfidf_object_part_paper_no_occ_selector(config):
    """
    Select two object parts, one with a high and one with a low tfidf.
    """
    if "ade_frequency" in config["Other"]:
        #tfidfs = get_ade_scenes_frequencies_raw_names(obj_first=False)
        raise("Not implemented")
    else:
        tfidfs, obj_freq = get_ade_tfidf_obj_part(filter_animacy=True, filter_plurals=True)
    pairs = []

    #obj_freq = sorted(list(obj_freq.items()), reverse=True, key=lambda t:t[1])
    #most_common = [t[0] for t in obj_freq[:50]]
    obj_freq_filtered = dict()
    for obj in obj_freq:
        if obj in tfidfs and len(tfidfs[obj]) >= 3:
            obj_freq_filtered[obj] = obj_freq[obj]

    obj_freq = sorted(list(obj_freq_filtered.items()), reverse=True, key=lambda t:t[1])
    most_common = [t[0] for t in obj_freq[:50]]

    for obj in most_common:
        parts = tfidfs[obj].items()
        parts = sorted(parts, key=lambda t: t[1])
        non_zero = [o for o in parts if o[1] != 0]
        high_tfidf = [o[0] for o in non_zero[-3:]]
        zero = [o[0] for o in parts if o[1] == 0]
        low_tfidf = [random.choice(zero)]
        for high in high_tfidf:
            for low in low_tfidf:
                info = {"object": obj, "orig_part": high, "foil_part": low, "type": "object-part-no-occ"}
                pair = gu.FoilPair(None, None, info=info)
                pairs.append(pair)
    return pairs


def ade_tfidf_same_scene_no_occ_paper_selector(config):
    """
    Select two objects from a category,
    one with a high tfidf and one which didn't occur in the category.
    """
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_scenes_frequencies_raw_names(obj_first=False)
    else:
        tfidfs = get_ade_scenes_tfidf_raw_names(obj_first=False, filter_animacy=True, filter_plurals=True)
    pairs = []

    for cat in tfidfs:
        if cat != "home_or_hotel":
            continue
        for sce in tfidfs[cat]:
            if sce.startswith("outliers"):
                continue
            objs = tfidfs[cat][sce].items()
            objs = sorted(objs, key=lambda t: t[1])
            non_zero = [o for o in objs if o[1] != 0]
            #avg = np.mean([o[1] for o in non_zero])
            #above = [o[0] for o in non_zero if o[1] > avg]
            #high_tfidf = above[-35:]
            high_tfidf = [o[0] for o in non_zero[-3:]]

            zero = [o[0] for o in objs if o[1] == 0]
            #print(len(zero))
            zero = random.choice(zero)
            #print(zero)
            #zero = [zero]

            for high in high_tfidf:
                #for low in zero:
                if "use_ade_cat" in config["Other"]:
                    info = {"category": cat, "outer_cat": cat, "orig_obj": high, "foil_obj": zero, "type": "scene-object-no-occ"}
                else:
                    info = {"category": sce, "outer_cat": cat, "orig_obj": high, "foil_obj": zero, "type": "scene-object-no-occ"}
                pair = gu.FoilPair(None, None, info=info)
                pairs.append(pair)
    return pairs



def ade_tfidf_same_scene_selector(config):
    """
    Select two objects from a scene type,
    one with a high tfidf or frequency and one with a low one.
    """
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_scenes_frequencies_raw_names(obj_first=False)
    else:
        tfidfs = get_ade_scenes_tfidf_raw_names(obj_first=False)
    pairs = []

    for cat in tfidfs:
        if cat == "unclassified":
            continue
        for sce in tfidfs[cat]:
            if sce.startswith("outliers"):
                continue
            objs = tfidfs[cat][sce].items()
            objs = sorted(objs, key=lambda t: t[1])
            non_zero = [o for o in objs if o[1] != 0]
            avg = np.mean([o[1] for o in non_zero])
            above = [o[0] for o in non_zero if o[1] > avg]
            below = [o[0] for o in non_zero if o[1] < avg]
            high_tfidf = above[-35:] #35 chosen because that gives a good number of examples
            low_tfidf = below
            for high in high_tfidf:
                for low in low_tfidf:
                    if "use_ade_cat" in config["Other"]:
                        info = {"category": cat, "outer_cat": cat, "orig_obj": high, "foil_obj": low}
                    else:
                        info = {"category": sce, "outer_cat": cat, "orig_obj": high, "foil_obj": low}
                    pair = gu.FoilPair(None, None, info=info)
                    pairs.append(pair)
    return pairs


def ade_tfidf_same_category_no_occ_selector(config):
    """
    Select two objects from a category,
    one with a high tfidf and one which didn't occur in the category.
    """
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_frequencies_raw_names(obj_first=False)
    else:
        tfidfs = get_ade_tfidf_raw_names(print_tfidfs=False, obj_first=False)
    pairs = []

    for cat in tfidfs:
        if cat == "unclassified":
            continue
        objs = tfidfs[cat].items()
        objs = sorted(objs, key=lambda t: t[1])
        non_zero = [o for o in objs if o[1] != 0]
        avg = np.mean([o[1] for o in non_zero])
        above = [o[0] for o in non_zero if o[1] > avg]
        high_tfidf = above[-35:]

        zero = [o[0] for o in objs if o[1] == 0][:35]

        for high in high_tfidf:
            for low in zero:
                info = {"category": cat, "orig_obj": high, "foil_obj": low}
                pair = gu.FoilPair(None, None, info=info)
                pairs.append(pair)
    return pairs


def ade_tfidf_same_scene_no_occ_selector(config):
    """
    Select two objects from a category,
    one with a high tfidf and one which didn't occur in the category.
    """
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_scenes_frequencies_raw_names(obj_first=False)
    else:
        tfidfs = get_ade_scenes_tfidf_raw_names(obj_first=False)
    pairs = []

    for cat in tfidfs:
        if cat == "unclassified":
            continue
        for sce in tfidfs[cat]:
            if sce.startswith("outliers"):
                continue
            objs = tfidfs[cat][sce].items()
            objs = sorted(objs, key=lambda t: t[1])
            non_zero = [o for o in objs if o[1] != 0]
            avg = np.mean([o[1] for o in non_zero])
            above = [o[0] for o in non_zero if o[1] > avg]
            high_tfidf = above[-35:]

            zero = [o[0] for o in objs if o[1] == 0][:35]

            for high in high_tfidf:
                for low in zero:
                    if "use_ade_cat" in config["Other"]:
                        info = {"category": cat, "outer_cat": cat, "orig_obj": high, "foil_obj": low}
                    else:
                        info = {"category": sce, "outer_cat": cat, "orig_obj": high, "foil_obj": low}
                    pair = gu.FoilPair(None, None, info=info)
                    pairs.append(pair)
    return pairs


def ade_tfidf_same_object_selector(config):
    """
    Select two categories where the object has a high tfidf
    in one category and a low one in the other.
    """
    ade_path = config["Datasets"]["ade_path"]
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_frequencies_raw_names()
    else:
        tfidfs = get_ade_tfidf_raw_names(print_tfidfs=False)
    pairs = []

    for obj_name in tfidfs:
        cat_tfs = tfidfs[obj_name]
        obj_name = gu.remove_accents(obj_name)
        cat_tfs = sorted(cat_tfs.items(), key=lambda t: t[1], reverse=True)
        cat_tfs = [ct for ct in cat_tfs if ct[0] != "unclassified"]
        non_zero = [ct for ct in cat_tfs if ct[1] != 0]
        avg = np.mean([ct[1] for ct in non_zero])
        orig = [ct[0] for ct in non_zero if ct[1] >= avg]
        foils = [ct[0] for ct in non_zero if ct[1] < avg]
        for orig_cat in orig:
            for foil_cat in foils:
                info = {"obj": obj_name, "orig_cat": orig_cat, "foil_cat": foil_cat}
                pair = gu.FoilPair(None, None, info=info)
                pairs.append(pair)

    return pairs


def ade_tfidf_same_object_scenes_selector(config):
    """
    Select two scene types where the object has a high tfidf
    in one scene type and a low one in the other.
    """
    ade_path = config["Datasets"]["ade_path"]
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_scenes_frequencies_raw_names()
    else:
        tfidfs = get_ade_scenes_tfidf_raw_names()
    pairs = []

    for cat in tfidfs:
        if cat == "unclassified":
            continue
        for obj_name in tfidfs[cat]:
            sce_tfs = tfidfs[cat][obj_name]
            obj_name = gu.remove_accents(obj_name)
            sce_tfs = sorted(sce_tfs.items(), key=lambda t: t[1], reverse=True)
            sce_tfs = [ct for ct in sce_tfs if not ct[0].startswith("outliers")]
            non_zero = [ct for ct in sce_tfs if ct[1] != 0]
            if len(non_zero) < 2:
                continue
            avg = np.mean([ct[1] for ct in non_zero])
            orig = [ct[0] for ct in non_zero if ct[1] >= avg]
            foils = [ct[0] for ct in non_zero if ct[1] < avg]
            for orig_sce in orig:
                for foil_sce in foils:
                    info = {"obj": obj_name, "orig_cat": orig_sce, "foil_cat": foil_sce, "category": cat}
                    pair = gu.FoilPair(None, None, info=info)
                    pairs.append(pair)

    return pairs


def ade_tfidf_same_object_no_occ_selector(config):
    """
    Select two categories where the object has a high tfidf
    in one category and didn't occurr in the other.
    """
    ade_path = config["Datasets"]["ade_path"]
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_frequencies_raw_names()
    else:
        tfidfs = get_ade_tfidf_raw_names(print_tfidfs=False)
    pairs = []

    for obj_name in tfidfs:
        cat_tfs = tfidfs[obj_name]
        obj_name = gu.remove_accents(obj_name)
        cat_tfs = sorted(cat_tfs.items(), key=lambda t: t[1], reverse=True)
        cat_tfs = [ct for ct in cat_tfs if ct[0] != "unclassified"]
        non_zero = [ct for ct in cat_tfs if ct[1] != 0]
        if len(non_zero) == 0:
            continue
        zero = [ct[0] for ct in cat_tfs if ct[1] == 0]
        avg = np.mean([ct[1] for ct in non_zero])
        orig_examples = [ct[0] for ct in non_zero if ct[1] >= avg]
        for orig_cat in orig_examples:
            for foil_cat in zero:
                info = {"obj": obj_name, "orig_cat": orig_cat, "foil_cat": foil_cat}
                pair = gu.FoilPair(None, None, info=info)
                pairs.append(pair)

    return pairs


def ade_tfidf_same_object_scenes_no_occ_selector(config):
    """
    Select two scene types where the object has a high tfidf
    in one scene type and didn't occurr in the other.
    """
    ade_path = config["Datasets"]["ade_path"]
    if "ade_frequency" in config["Other"]:
        tfidfs = get_ade_scenes_frequencies_raw_names()
    else:
        tfidfs = get_ade_scenes_tfidf_raw_names()
    pairs = []

    for cat in tfidfs:
        if cat == "unclassified":
            continue
        for obj_name in tfidfs[cat]:
            sce_tfs = tfidfs[cat][obj_name]
            obj_name = gu.remove_accents(obj_name)
            sce_tfs = sorted(sce_tfs.items(), key=lambda t: t[1], reverse=True)
            sce_tfs = [ct for ct in sce_tfs if ct[0] != "outliers"]
            non_zero = [ct for ct in sce_tfs if ct[1] != 0]
            if len(non_zero) == 0:
                continue
            zero = [ct[0] for ct in sce_tfs if ct[1] == 0]
            avg = np.mean([ct[1] for ct in non_zero])
            orig_examples = [ct[0] for ct in non_zero if ct[1] >= avg]
            for orig_sce in orig_examples:
                for foil_sce in zero:
                    info = {"obj": obj_name, "orig_cat": orig_sce, "foil_cat": foil_sce, "category": cat}
                    pair = gu.FoilPair(None, None, info=info)
                    pairs.append(pair)

    return pairs


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

    num_examples = 20000

    train_path = os.path.join(ade_path, "images/ADE/training/")
    categories = set(os.listdir(train_path))

    pairs = []
    fns = list(enumerate(index['filename']))
    selected_inds = set()
    while len(selected_inds) < num_examples:
        i, fn = random.choice(fns)
        if i in selected_inds:
            continue
        selected_inds.add(i)
        path = index["folder"][i]
        category = path.split("/")[-2]
        if category == "unclassified":
            continue
        cat_paths = enumerate(index['folder'])
        new_cat_paths = [j for (j,p) in cat_paths if not category in p]
        j = random.choice(new_cat_paths)
        foil_fn = index['filename'][j]
        pair = gu.FoilPair(i, j)
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

    num_examples = 20000
    categories = enumerate([c for c in index['folder'] if not "unclassified" in c])
    inds = [i for (i,c) in categories]
    train_fns = [(i,fn) for (i,fn) in enumerate(index['filename']) if i in inds]
    selected_fns = random.choices(train_fns, k=num_examples)

    train_path = os.path.join(ade_path, "images/ADE/training/")

    pairs = []
    for i, fn in selected_fns:
        path = index["folder"][i]
        category = path.split("/")[-2]
        scene = path.split("/")[-1]
        cat_path = os.path.join(train_path, category)
        scenes = [d for d in os.listdir(cat_path) if d!=scene]
        scene_ids = list(enumerate(index['scene']))
        same_cat = [(j,s) for (j,s) in scene_ids if s in scenes]
        foil_ids = [j for (j,_) in same_cat]
        j  = random.choice(foil_ids)
        foil_fn = index['filename'][j]
        pair = gu.FoilPair(i, j)
        pairs.append(pair)
    return pairs


def ade_same_image_selector(config):
    """
    Select one image and use it as both original and foil.

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

    num_examples = 20000
    categories = enumerate([c for c in index['folder'] if not "unclassified" in c])
    inds = [i for (i,c) in categories]
    train_fns = [(i,fn) for (i,fn) in enumerate(index['filename']) if i in inds]
    selected_fns = random.choices(train_fns, k=num_examples)

    train_path = os.path.join(ade_path, "images/ADE/training/")

    pairs = []
    for i, fn in selected_fns:
        pair = gu.FoilPair(i, i)
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
    if config["General"]["similar"] == "True":
        similar = True
    else:
        similar = False
    cxc_subset = config["Datasets"]["cxc_subset"]
    conditions = config["General"]["conditions"].split("\n")
    if conditions[0] != '':
        cond_fns = [getattr(sc, cond) for cond in conditions]
    else:
        cond_fns = []

    similarities = gu.read_cxc(cxc_path, cxc_subset)

    if similar:
        print("higher")
        idpairs = [idpair for idpair in similarities if similarities[idpair] > cutoff]
    else:
        print("lower")
        idpairs = [idpair for idpair in similarities if similarities[idpair] < cutoff]

    l = [p for p in idpairs if p[0]==224802]
    coco2vg = gu.get_vg_image_ids(config, reverse=True)
    l = [(i1, coco2vg[i2] if i2 in coco2vg else None) for (i1,i2) in l]
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

    for idpair in idpairs:
        sort_out = False
        for cond_fn in cond_fns:
            if cond_fn(idpair, config, info=info) == False:
                sort_out = True
                break

        if sort_out == True:
            continue

        i1id, i2id = idpair
        if config["Functions"]["generator"] == "caption_pair_generator":
            # for the captions, we can use coco images
            # not contained in VisualGenome
            pass
        elif config["Functions"]["generator"] != "caption_adj_generator":
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


def cxc_same_selector(config):
    """
    Select an image as both original and foil image.
    Only use images annotated in CxC, 
    to keep the pair comparable to the ones produced
    by cxc_similar_selector.

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
    if "debug" in config["Other"] and config["Other"]["debug"] == True:
        similarities = similarities[:20]

    idpairs = set()
    for idpair in similarities:
        i1id, i2id = idpair
        idpairs.add((i1id,i1id))
        idpairs.add((i2id,i2id))

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

    for idpair in idpairs:
        sort_out = False
        for cond_fn in cond_fns:
            if cond_fn(idpair, config, info=info) == False:
                sort_out = True
                break

        if sort_out == True:
            continue

        i1id, i2id = idpair
        if config["Functions"]["generator"] == "caption_pair_generator":
            # for the captions, we can use coco images
            # not contained in VisualGenome
            pass
        elif config["Functions"]["generator"] == "caption_adj_generator":
            # here, the foil image still needs to be in VG, though
            i2id = coco2vg[i2id]
        else:
            # using VG ids is usually more useful
            i1id = coco2vg[i1id]
            i2id = coco2vg[i2id]
        pair = gu.FoilPair(i1id, i2id)
        pairs.append(pair)

    return pairs
