import os
import random
import json
import csv

random.seed(0)


class FoilPair:
    """
    A pair of two text snippets, where in one of the snippets, 
    some part of the text was replaced by foil text.

    Attributes
    ----------
    context : N-tuple of strings
        Basic context produced by context_generator,
        into which the correct and foil words are to be inserted.
    orig_img : str
        Path to annotations or image id of original image.
    foil_img : str
        Path to annotations or image id of image chosen to select foil word.
    correct : dict
        Condition in syntaxgym suite format containing the correct text.
    foiled : dict
        Condition in syntaxgym suite format containing the foiled text.
    region_meta : dict
        Region names in the format required by the sntaxgym json representation.
    formula : str
        Formula for syntaxgym to determine result for pair.
    info : dict
        Any further information to be passed between
        the parts of the suite generation pipeline.
    """
    def __init__(self, context, orig_img, info=None):
        self.context = context
        self.orig_img = orig_img
        self.foil_img = None

        self.correct = dict()
        self.correct["condition_name"] = "correct"
        self.correct["regions"] = []
        self.foiled = dict()
        self.foiled["condition_name"] = "foiled"
        self.foiled["regions"] = []

        self.region_meta = None
        self.formula = None

        self.info = None

def get_ade_paths(ade_base_path):
    """
    Get paths of annotations of the individual images in ADE20k.

    Parameters
    ----------
    ade_base_path : str
        Path to ADE20k data set.
    """

    environments = os.listdir(ade_base_path)
    env_paths = [os.path.join(ade_base_path, e) for e in environments]
    specific_env_paths = []
    for ep in env_paths:
        specific_envs = os.listdir(ep)
        specific_env_paths += [os.path.join(ep, spe) for spe in specific_envs]
    img_info_paths = []
    for sep in specific_env_paths:
        img_info_paths += [os.path.join(sep, ip) for ip in os.listdir(sep) if ip.endswith("json")]

    return img_info_paths

def get_vg_image_ids(vg_base_path, mscoco_path, reverse=False):
    """
    Get Visual Genome image ids for images
    that also have a coco_id, for use with CxC.
    Only use those whose id is in the val2014 set,
    since I only have val annotations for both
    CxC and MSCOCO.

    Parameters
    ----------
    vg_base_path : str
        Path to Visual Genome dataset.
    mscoco_path : str
        Path to MSCOCO caption annotations.
    reverse : bool
        If true, give dict with coco_ids as keys.

    Return
    ------
    vg2coco : dict
        Dictionary mapping each VG id to an MSCOCO id.
    """

    with open(os.path.join(vg_base_path, "image_data.json")) as f:
        image_data = json.loads(f.read())

    with open(os.path.join(mscoco_path, "instances_val2014.json")) as cf:
        coco_data = json.loads(cf.read())
        coco_ids = set([c['id'] for c in coco_data])

    with_coco = [i for i in image_data if i['coco_id'] in coco_ids]

    if not reverse:
        vg2coco = {i['image_id']: i['coco_id'] for i in with_coco}
    else:
        #inconsistent naming
        vg2coco = {i['coco_id']: i['image_id'] for i in with_coco}

    return vg2coco


def _coco_fn2img_id(coco_fn):
    without_pref = coco_fn.split("_")[-1]
    without_ending = without_pref.split(".")[0]
    while without_ending[0] == '0':
        without_ending = without_ending[1:]
    return without_ending

def read_cxc(cxc_path, filename):
    """
    Read information from CxC dataset into a dict.

    Parameters
    ----------
    cxc_path : str
        Path to CxC dataset.
    filename : str
        Specific file in CxC to use.

    Return
    ------
    similarities: dict(str, dict(str, float))
        Dictionary of similarity pairs. For each image_id,
        contains a dictionary from the other image_ids for which
        there is a similarity judgement to the similarity score.
    """
    
    similarities = dict()
    with open(os.path.join(cxc_path, filename), newline='') as cxcfile:
        cxcreader = csv.reader(cxcfile, delimiter=',')
        for i, row in enumerate(cxcreader):
            if i == 0:
                continue
            img_id1 = _coco_fn2img_id(row[0])
            img_id2 = _coco_fn2img_id(row[1])
            similarity = float(row[2])

            if not img_id1 in similarities:
                similarities[img_id1] = dict()
            if not img_id2 in similarities:
                similarities[img_id2] = dict()

            similarities[img_id1][img_id2] = similarity
            similarities[img_id2][img_id1] = similarity

    return similarities
