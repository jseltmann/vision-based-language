import os
import random
import configparser
from tqdm import tqdm
import json

import generation_utils as gu
import selection_conditions as sc


random.seed(0)

#def ade_same_env_selector(pairs):
#    """
#    Select an image from the same ADE20k category as foil image.
#
#    Parameters
#    ----------
#    pairs : [FoilPair]
#        List of foil pairs without foil images.
#
#    Returns
#    -------
#    pairs_with_foil : [FoilPair]
#        Pairs with selected foil image.
#    """
#
#    pairs_with_foil = []
#    for pair in pairs:
#        orig_path = pair.orig_img
#        orig_fn = os.path.basename(orig_path)
#        category_path = os.path.dirname(orig_path)
#        
#        imgs = os.listdir(category_path)
#        imgs = [i for i in imgs if i.endswith("json")]
#        imgs.remove(orig_fn)
#        foil_img = random.choice(imgs)
#        foil_path = os.path.join(category_path, foil_img)
#
#        pair.foil_img = foil_path
#        pairs_with_foil.append(pair)
#
#    return pairs_with_foil
#
#def cxc_sis_similar_selector(pairs):
#    """
#    Select the foil image to be similar to the original
#    based on the the CxC dataset. Requires the original
#    image to be part of MSCOCO 2014 val or train sets.
#    """
#    cxc_path = "/home/jseltmann/data/Crisscrossed-Captions/data"
#    
#    similarities = read_cxc(cxc_path, "sis_val.csv")

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
    num_examples = int(config["General"]["num_examples"])
    cutoff = float(config["General"]["cutoff"])
    similar = bool(config["General"]["similar"])
    cxc_subset = config["Datasets"]["cxc_subset"]
    conditions = config["General"]["conditions"].split("\n")
    cond_fns = [getattr(sc, cond) for cond in conditions]

    similarities = gu.read_cxc(cxc_path, cxc_subset)

    if similar:
        idpairs = [idpair for idpair in similarities if similarities[idpair] > cutoff]
    else:
        idpairs = [idpair for idpair in similarities if similarities[idpair] < cutoff]

    pairs = []
    chosen = set()

    num_tries = 0
    found = 0

    vg2coco = gu.get_vg_image_ids(config)
    attrs_as_dict = gu.attrs_as_dict(config)
    coco_dict = gu.coco_as_dict(config)

    info = {'attrs': attrs_as_dict,
            'vg2coco': vg2coco,
            'coco_captions': coco_dict}

    for idpair in tqdm(idpairs):
        sort_out = False
        for cond_fn in cond_fns:
            if cond_fn(idpair, config, info=info) == False:
                sort_out = True
                break

        if sort_out == True:
            continue
        found += 1

        i1id, i2id = idpair
        pair = gu.FoilPair(i1id, i2id)
        pairs.append(pair)

    return pairs
