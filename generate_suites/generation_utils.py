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
    def __init__(self, orig_img, foil_img, info={}):
        self.context = None
        self.orig_img = orig_img
        self.foil_img = foil_img

        self.correct = dict()
        self.correct["condition_name"] = "correct"
        self.correct["regions"] = []
        self.foiled = dict()
        self.foiled["condition_name"] = "foiled"
        self.foiled["regions"] = []

        self.region_meta = None
        self.formula = None

        self.info = info


def get_vg_image_ids(config, reverse=False):
    """
    Get Visual Genome image ids for images
    that also have a coco_id, for use with CxC.
    Only use those whose id is in the val2014 set,
    since I only have val annotations for both
    CxC and MSCOCO.

    Parameters
    ----------
    config : Configparser
        Read from config file.
    reverse : bool
        If true, give dict with coco_ids as keys.

    Return
    ------
    vg2coco : dict
        Dictionary mapping each VG id to an MSCOCO id.
    """

    vg_base_path = config["Datasets"]["vg_path"]
    mscoco_path = config["Datasets"]["mscoco_path"]

    with open(os.path.join(vg_base_path, "image_data.json")) as f:
        image_data = json.loads(f.read())

    with open(os.path.join(mscoco_path, "instances_val2014.json")) as cf:
        coco_data = json.loads(cf.read())
        coco_ids = set([c['id'] for c in coco_data['annotations']])

    with_coco = [i for i in image_data if i['coco_id'] in coco_ids]

    if not reverse:
        vg2coco = {i['image_id']: i['coco_id'] for i in with_coco}
    else:
        #inconsistent naming
        vg2coco = {i['coco_id']: i['image_id'] for i in with_coco}

    return vg2coco


def ade_fn2index(fn):
    """
    Split an ADE filename to get the index
    of the file in the lists of the index file.
    """
    fn = fn.split(".")[0]
    number = fn.split("_")[-1]
    number = int(number) - 1
    return number


def get_ade_json_path(fn, data_path, index):
    orig_id = ade_fn2index(fn)
    orig_dir = os.path.join(data_path, index["folder"][orig_id])
    json_name = fn.split(".")[0] + ".json"
    json_path = os.path.join(orig_dir, json_name)
    return json_path


def _coco_fn2img_id(coco_fn):
    without_pref = coco_fn.split("_")[-1]
    without_ending = without_pref.split(".")[0]
    while without_ending[0] == '0':
        without_ending = without_ending[1:]
    image_id = int(without_ending)
    return image_id

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
    similarities: dict((str,str), float))
        Dictionary of of pairs of image ids and similarities
        between the images in the pair.
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

            similarities[(img_id1,img_id2)] = similarity

    return similarities


def vg_as_dict(config, filename, keys="coco"):
    """
    Get specific Visual Genome file as a dict,
    with MSCoco or Visual Genome image ids as keys and the
    attribute information for each image as values.
    """

    vg_path = config["Datasets"]["vg_path"]
    if keys == "coco":
        vg2coco = get_vg_image_ids(config)
    if filename.endswith("json"):
        with open(os.path.join(vg_path, filename)) as rel_file:
            info = json.loads(rel_file.read())
    else:
        with open(os.path.join(vg_path, filename+".json")) as rel_file:
            info = json.loads(rel_file.read())
    vg_as_dict = dict()
    for img in info:
        if keys == "coco":
            if not img['image_id'] in vg2coco:
                continue
            cocoid = vg2coco[img['image_id']]
            vg_as_dict[cocoid] = img
        else:
            vg_as_dict[img['image_id']] = img

    return vg_as_dict


def coco_as_dict(config):
    """
    Get MSCOCO captions as dict
    with image_ids as keys.
    """
    coco_path = config["Datasets"]["mscoco_path"]
    with open(os.path.join(coco_path, "captions_val2014.json")) as cf:
        caption_data = json.loads(cf.read())['annotations']
    coco_dict = dict()
    for img in caption_data:
        coco_dict[img['image_id']] = img['caption']
    return coco_dict
