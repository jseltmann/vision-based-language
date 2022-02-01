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
from collections import defaultdict

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
        if config["Functions"]["selector"] == "ade_different_category_selector":
            scene = json_path.split("/")[-3]
        else:
            scene = json_path.split("/")[-2]
        #scene = random.choice(annot['scene'])
        #doc = nlp(scene)
        #if doc[0].pos_ == "NOUN":
        #    context = ("This is " + p.a(scene) + ".",)
        #else:
        #    context = ("This is " + p.a(scene) + " place.",)
        context = (gu.ade_cat2text(scene),)
        if context[0] is None:
            context = ("This is an unclassified place.",)

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
            new_pair.formula = "(2;%foiled%) > (2;%correct%)"
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
            context = ("There is", word)
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
                subj = rel['subject']['name'].strip('"')
            else:
                subj = rel['subject']['names'][0].strip('"')
            pred = rel['predicate'].lower().strip('"')
            context = (subj + " " + pred,)
            new_pair = copy.deepcopy(pair)
            new_pair.context = context
            if 'name' in rel['object']:
                new_pair.info = {"orig_obj": rel['object']['name'].strip('"')}
            else:
                new_pair.info = {"orig_obj": rel['object']['names'][0].strip('"')}
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
        pair.info["context_cap_obj"] = img
        pair.info["2nd_cap_object"] = None
        cap = ' '.join(img['caption'].split())
        doc = nlp(cap)
        adj_positions = [i for i, word in enumerate(doc) if word.pos_=='ADJ']
        adj_ = [word for word in doc if word.pos_=='ADJ']

        adj_pos = random.choice(adj_positions)
        earlier = doc[:adj_pos]
        if len(earlier) == 0:
            info =  {'indefinite' : False, 'start' : True}
            article = None
        elif  'Ind' in earlier[-1].morph.get('Definite'):
            # remove indefinite article,
            # since we don't know what sound the inserted word starts with
            article = earlier[-1].text
            earlier = earlier[:-1]
            info = {'indefinite': True}
        else:
            info = {'indefinite': False}
            article = None
        later = doc[adj_pos+1:]

        context = (earlier.text, later.text)
        pair.context = context
        pair.info = info
        pair.info["orig_adj"] = doc[adj_pos].text.strip()

        if article:
            r1 = (earlier.text + " " + article).strip()
        else:
            r1 = earlier.text
        pair.correct["regions"].append({"region_number":1, "content": r1})
        r2 = doc[adj_pos].text.strip()
        pair.correct["regions"].append({"region_number":2, "content": r2})
        r3 = later.text.strip()
        pair.correct["regions"].append({"region_number":3, "content": r3})

        pair.region_meta = {"1": "earlier", "2": "adj", "3": "later"}
        pair.formula = "(2;%foiled%) > (2;%correct%)"

    return pairs


def caption_generator(pairs, config):
    """
    Use a caption as context.

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
        caption_dict = defaultdict(list)
        for img in caption_data:
            caption_dict[img['image_id']].append(img['caption'])

    #vg2coco = gu.get_vg_image_ids(config)

    #for pair in tqdm(pairs):
    for pair in pairs:
        #img = caption_dict[vg2coco[pair.orig_img]]
        img = caption_dict[pair.orig_img]
        cap = ' '.join(img['caption'].split())

        pair.context = (cap,)
        pair.info = info

        r1 = earlier.text.strip()
        pair.correct["regions"].append({"region_number":1, "content": r1})
        r2 = doc[adj_pos].text.strip()
        pair.correct["regions"].append({"region_number":2, "content": r2})
        r3 = later.text.strip()
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
    nlp = spacy.load("en_core_web_sm")

    new_pairs = []
    for pair in pairs:
        orig_objs = objs[pair.orig_img]["objects"]
        #obj_names = list(set([o["names"][0] for o in orig_objs]))
        obj_names = [o["names"][0] for o in orig_objs]
        objs_per_example = 5
        while len(set(obj_names)) > objs_per_example:
            new_pair = copy.deepcopy(pair)
            #curr_objs = obj_names[:objs_per_example]
            #obj_names = obj_names[objs_per_example:]
            curr_objs = random.sample(obj_names, k=objs_per_example)
            for o in curr_objs:
                obj_names.remove(o)

            context_objs = curr_objs[:(objs_per_example-1)]
            context = "There is "
            for obj in context_objs:
                obj_nlp = nlp(obj)[-1]
                if obj_nlp.lemma_ == obj_nlp.text: # singular
                    context += p.a(obj) + ", "
                else:
                    context += obj + ", "
            context += "and"
            new_pair.context = context
            new_pair.info = {"corr_obj": obj_names[-1]}

            new_pairs.append(new_pair)

    return new_pairs


def pass_generator(pairs, config):
    """
    Pass pairs without generating anything.
    """
    return pairs


def caption_pair_generator(pairs, config):
    """
    Set caption as context to be extended.
    """
    
    coco_dict = gu.coco_as_dict_list(config)
    cxc_path = config["Datasets"]["cxc_path"]
    cap_sim = gu.read_cxc_sentence_similarities(cxc_path)

    if "cxc_caption_similarity" in config["Other"]:
        similarity_func = "cxc"
    else:
        similarity_func = "jaccard"
        pairs = gu.tokenize_caps(pairs, coco_dict)

    if "low_sim" in config["Other"]:
        sim_cutoff = 0.2
    else:
        sim_cutoff = 0.4

    new_pairs = []
    for pair in pairs:
        orig_img = pair.orig_img
        if similarity_func == "jaccard":
            #captions = coco_dict[orig_img["img_id"]]
            captions = coco_dict[orig_img]
            #captions = pair.info["captions_orig"]
            jaccs = gu.get_jaccard_similarities(captions)
            sims_low = [triple for triple in jaccs if triple[2] < sim_cutoff]
        else:
            #captions = coco_dict[orig_img["img_id"]]
            captions = coco_dict[orig_img]
            sims = gu.get_cxc_cap_similarities(cap_sim, captions)
            sims_low = [triple for triple in sims if triple[2] < sim_cutoff]
            #raise("CxC caption similarity not implemented.")

        for (c1, c2, sim) in sims_low:
            new_pair = copy.deepcopy(pair)
            new_pair.context = (c1["caption"],)
            new_pair.correct['regions'].append({"region_number":1, "content": c1["caption"]})
            new_pair.correct['regions'].append({"region_number":2, "content": c2["caption"]})
            new_pair.region_meta = {"1": "caption1", "2": "caption2"}
            new_pair.formula = "(2;%foiled%) > (2;%correct%)"
            new_pair.info["context_cap_obj"] = c1
            new_pairs.append(new_pair)

    return new_pairs
