import os
import random
import json
import csv
import inflect
import spacy
from collections import defaultdict
from tqdm import tqdm
import unicodedata

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

    def __str__(self):
        s = str(self.orig_img) + "\n" + str(self.foil_img) + "\n"
        s += "correct:\n"
        for r in self.correct["regions"]:
            s += r["content"] + "\n"
        s += "foiled:\n"
        for r in self.foiled["regions"]:
            s += r["content"] + "\n"
        return s


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


def get_ade_json_path(pos, data_path, index):
    orig_dir = os.path.join(data_path, index["folder"][pos])
    fn = index['filename'][pos].split(".")[0] + ".json"
    json_path = os.path.join(orig_dir, fn)
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


def read_cxc_sentence_similarities(cxc_path):
    """
    Read CxC similarities for captions into a dict.
    """
    cxc_dict = dict()
    with open(os.path.join(cxc_path, "sts_val.csv"), newline='') as cxcfile:
        cxcreader = csv.reader(cxcfile)
        for i, row in enumerate(cxcreader):
            if i == 0:
                continue
            c1id = int(row[0].split(":")[-1])
            c2id = int(row[1].split(":")[-1])
            sim = float(row[2])
            cxc_dict[(c1id,c2id)] = sim
            cxc_dict[(c2id,c1id)] = sim
    return cxc_dict


def get_cxc_cap_similarities(cxc_dict, caption_objs, ind_caption=None):
    """
    Retrieve CxC similarities for given coco captions.
    If ind_caption is not None, retrieve similarities between it and 
    each caption_obj. Otherwise, retrieve pairwise similarities between
    caption_objs.
    """
 
    sims = []
    if ind_caption:
        cap_tuples = [(ind_caption,b) for b in caption_objs]
    else:
        cap_tuples = [(a,b) for a in caption_objs for b in caption_objs]

    for (capa, capb) in cap_tuples:
        idpair = (capa['id'], capb['id'])
        if idpair in cxc_dict:
            sims.append((capa, capb, cxc_dict[idpair]))
    return sims


def vg_as_dict(config, filename, keys="coco"):
    """
    Get specific Visual Genome file as a dict,
    with MSCoco or Visual Genome image ids as keys and the
    attribute information for each image as values.
    """

    vg_path = config["Datasets"]["vg_path"]
    vg2coco = get_vg_image_ids(config)
    if filename.endswith("json"):
        with open(os.path.join(vg_path, filename)) as rel_file:
            info = json.loads(rel_file.read())
    else:
        with open(os.path.join(vg_path, filename+".json")) as rel_file:
            info = json.loads(rel_file.read())
    vg_as_dict = dict()
    for img in info:
        if 'image_id' in img:
            img_id = img['image_id']
        else:
            img_id = img['id']
        if not img_id in vg2coco: # only use images in coco-val
            continue
        if keys == "coco":
            if not img_id in vg2coco:
                continue
            cocoid = vg2coco[img_id]
            vg_as_dict[cocoid] = img
        else:
            vg_as_dict[img_id] = img

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


def coco_as_dict_list(config):
    """
    Get MSCOCO captions as dict
    with image_ids as keys.
    """
    coco_path = config["Datasets"]["mscoco_path"]
    with open(os.path.join(coco_path, "captions_val2014.json")) as cf:
        caption_data = json.loads(cf.read())['annotations']
    coco_dict = defaultdict(list)
    for img in caption_data:
        coco_dict[img['image_id']].append(img)
    return coco_dict



ade_cat2text_dict = {
    "transportation": "This is a transportation scene.",
    "cultural": "This is a cultural place.",
    "nature_landscape": "This is a place in nature.",
    "sports_and_leisure": "This is a place for sports and leisure.",
    "home_or_hotel": "This is a home or hotel.",
    "work_place": "This is a workplace.",
    "urban": "This is an urban place.",
    "industrial": "This is an industrial place.",
    "shopping_and_dining": "This is a place for shopping and dining.",
    "unclassified": None,
    "arena__hockey": "This is a hockey arena.",
    "stadium__baseball": "This is a baseball stadium.",
    "underwater__coral_reef": "This is a coral reef.",
    "arena__soccer": "This is a soccer arena."
}


def ade_cat2text(cat):
    if cat in ade_cat2text_dict:
        return ade_cat2text_dict[cat]
    words = cat.split("__") # remove extra information in scene names
    words = words[0]
    words = words.split("_")
    cat_name = " ".join(words)

    p = inflect.engine()
    sentence = "This is " + p.a(cat_name) + "."

    return sentence


def tokenize_caps(coco_dict):
    nlp = spacy.load("en_core_web_sm")
    stopwords = nlp.Defaults.stop_words
    tokenized_dict = dict()
    for img_id, captions in coco_dict.items():
        tokenized_caps = []
        for caption in captions:
            cap_text = [t.text.lower() for t in nlp.tokenizer(caption["caption"])]
            w1 = set(cap_text)
            w1 = {word for word in w1 if word not in stopwords}
            caption["tokenized"] = w1
            tokenized_caps.append(caption)
        tokenized_dict[img_id] = tokenized_caps
    return tokenized_dict


def get_jaccard_similarities(captions, ind_caption=None):
    """
    Get jaccard similarities between different captions.
    If ind_caption is not None, similarities will be calculated between
    it and each of the given captions.
    Otherwise, calculate the pairwise similarities of the captions.
    """
    if ind_caption:
        caps1 = [ind_caption]
    else:
        caps1 = captions
    cap_pairs = [(a,b) for a in caps1 for b in captions]

    jaccs = []
    for (cap1, cap2) in cap_pairs:
        w1 = cap1["tokenized"]
        w2 = cap2["tokenized"]
        inter = [w for w in w1 if w in w2]
        union = w1.union(w2)
        jacc = len(inter) / len(union)
        jaccs.append((cap1, cap2, jacc))

    return jaccs


def add_caption_context(pairs, config):
    """
    Add image captions as context.
    """

    if config["Functions"]["generator"].startswith("caption"):
        coco_keys = True
    else:
        coco_keys = False
        vg2coco = get_vg_image_ids(config)

    captions = coco_as_dict_list(config)

    for pair in pairs:
        origid = pair.orig_img
        if not coco_keys:
            origid = vg2coco[origid]
        curr_captions = captions[origid]
        if "context_cap_obj" in pair.info:
            used_captions = [pair.info["context_cap_obj"], pair.info["2nd_cap_object"]]
            curr_captions = [c for c in curr_captions if c not in used_captions]
        chosen = random.choice(curr_captions)
        cap = chosen["caption"]
        cap = cap.replace('"', '')
        cap = " ".join([w for w in cap.split() if len(w) > 0])

        new_region = {"region_number": 0, "content": cap}
        pair.correct['regions'] = [new_region] + pair.correct['regions']

        pair.foiled['regions'] = [new_region] + pair.foiled['regions']
        pair.region_meta["0"] = "context_caption"

    return pairs


def remove_accents(text):
    text = ''.join([c for c in unicodedata.normalize('NFD', text)
                    if unicodedata.category(c) != 'Mn'])
    return text
