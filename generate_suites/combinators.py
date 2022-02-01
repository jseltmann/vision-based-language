import os
import random
import json
import inflect
import copy
import pickle
import codecs
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.similarities.annoy import AnnoyIndexer
import spacy

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
    nlp = spacy.load("en_core_web_sm")
    for i, pair in enumerate(pairs):
        json_path = gu.get_ade_json_path(pair.orig_img, data_path, index)
        with codecs.open(json_path, "r", "ISO-8859-1") as jfile:
            orig_annot = json.load(jfile)['annotation']

        orig_obj = random.choice(orig_annot['object'])
        orig_obj_name = orig_obj['raw_name']
        #correct_text = "There is " + p.a(orig_obj_name) + "."
        #if nlp(orig_obj_name)[0].lemma_ == orig_obj:
        #print(nlp(orig_obj_name)[-1].lemma_, nlp(orig_obj_name)[-1].text)
        #if i == 50:
        #    9 / 0
        obj_nlp = nlp(orig_obj_name)[-1]
        if obj_nlp.lemma_ == obj_nlp.text:
            singular = True
        else:
            singular = False
        if singular:
            article = p.a(orig_obj_name).split()[0]
            #correct_text = "There is " + p.a(orig_obj_name) + "."
            correct_text = "There is " + article
        else:
            #correct_text = "There are " + orig_obj_name + "."
            correct_text = "There are" #+ orig_obj_name + "."
        earlier = pair.context[0] + " " + correct_text
        #pair.correct["regions"].append({"region_number":1, "content":pair.context[0]})
        #pair.correct["regions"].append({"region_number":2, "content":correct_text})
        pair.correct["regions"].append({"region_number":1, "content":earlier})
        pair.correct["regions"].append({"region_number":2, "content":orig_obj_name})
        pair.correct["regions"].append({"region_number":3, "content":"."})

        json_path = gu.get_ade_json_path(pair.foil_img, data_path, index)
        with codecs.open(json_path, "r", "ISO-8859-1") as jfile:
            foil_annot = json.load(jfile)['annotation']
        foil_obj = random.choice(foil_annot['object'])
        foil_obj_name = foil_obj['raw_name']
        #foiled_text = "There is " + p.a(foil_obj_name) + "."
        #if nlp(foil_obj_name)[0].lemma_ == orig_obj:
        obj_nlp = nlp(foil_obj_name)[-1]
        if obj_nlp.lemma_ == obj_nlp.text:
        #if nlp(foil_obj_name)[-1].lemma_ == foil_obj_name.split()[-1]:
            singular = True
        else:
            singular = False
        #if singular:
        #    foiled_text = "There is " + p.a(foil_obj_name) + "."
        #else:
        #    foiled_text = "There are " + foil_obj_name + "."
        #pair.foiled["regions"].append({"region_number":1, "content":pair.context[0]})
        #pair.foiled["regions"].append({"region_number":2, "content":foiled_text})

        if singular:
            article = p.a(foil_obj_name).split()[0]
            correct_text = "There is " + article
        else:
            #correct_text = "There are " + orig_obj_name + "."
            correct_text = "There are" #+ orig_obj_name + "."
        earlier = pair.context[0] + " " + correct_text
        #pair.correct["regions"].append({"region_number":1, "content":pair.context[0]})
        #pair.correct["regions"].append({"region_number":2, "content":correct_text})
        pair.foiled["regions"].append({"region_number":1, "content":earlier})
        pair.foiled["regions"].append({"region_number":2, "content":foil_obj_name})
        pair.foiled["regions"].append({"region_number":3, "content":"."})

        pair.region_meta = {"1": "context", "2": "object", "3": "end"}
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
        #attr = attr.strip()
        attr = " ".join(attr.split()) # remove multiple spaces in attribute
        #r = pair.context[0] + p.a(attr)
        #r += " " + pair.context[1] + "."
        #r = r.strip()
        ##r += pair.context[1] + "."
        #pair.correct['regions'].append({"region_number":1, "content": r})
        article = p.a(attr).split()[0]
        earlier = pair.context[0] + " " + article
        earlier = earlier.strip()
        pair.correct['regions'].append({"region_number":1, "content": earlier})
        pair.correct['regions'].append({"region_number":2, "content": attr})
        later = pair.context[1].strip() + "."
        pair.correct['regions'].append({"region_number":3, "content": later})

        img_attrs = attrs[pair.foil_img]['attributes']
        has_attrs = lambda o: 'attributes' in o.keys() and o['attributes'] != []
        objs = [obj for obj in img_attrs if has_attrs(obj)][:10]
        for obj in objs:
            #obj = random.choice(objs)
            attr = random.choice(obj['attributes'])
            attr = " ".join(attr.split()) # remove multiple spaces in attribute
            #attr = attr.strip()

            #r = pair.context[0] + p.a(attr)
            #r += " " + pair.context[1] + "."
            #r = r.strip()
            new_pair = copy.deepcopy(pair)
            #new_pair.foiled['regions'].append({"region_number":1, "content": r})
            article = p.a(attr).split()[0]
            earlier = pair.context[0] + " " + article
            earlier = earlier.strip()
            new_pair.foiled['regions'].append({"region_number":1, "content": earlier})
            new_pair.foiled['regions'].append({"region_number":2, "content": attr})
            #later = pair.context[1].strip() + "."
            new_pair.foiled['regions'].append({"region_number":3, "content": later})

            new_pair.region_meta = {"1": "context", "2": "attribute", "3": "object"}
            new_pair.formula = "(2;%foiled%) > (2;%correct%)"
            new_pairs.append(new_pair)

    return new_pairs


def caption_adj_combinator(pairs, config):
    """
    Generate foil text by replacing an adjective
    with an attribute from a different context.
    """

    attrs = gu.vg_as_dict(config, "attributes", keys="visgen")
    p = inflect.engine()

    full_pairs = []
    for pair in pairs:
    #for pair in tqdm(pairs):
        img_attrs = attrs[pair.foil_img]['attributes']
        has_attrs = lambda o: 'attributes' in o.keys() and o['attributes'] != []
        objs = [obj for obj in img_attrs if has_attrs(obj)][:5]
        for obj in objs:
            new_pair = copy.deepcopy(pair)
            attr = random.choice(obj['attributes'])
            if attr.strip() == "":
                continue
            attr = " ".join(attr.split())

            earlier, later = new_pair.context
            #if "young child" in earlier or "tennis ball" in later:
            #    print(earlier, later, attr)
            if 'start' in new_pair.info and new_pair.info['start'] == True:
                new_pair.foiled["regions"].append({"region_number":1, "content": ""})
                r2 = attr.capitalize().strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            elif new_pair.info['indefinite']:
                if len(earlier) > 0:
                    r1 = earlier + " " + p.a(attr).strip()
                else:
                    r1 = p.a(attr).strip()
                r1 = " ".join(r1.split()[:-1])
                #r1 = earlier + " " + p.a(attr).strip()
                r1 = r1.strip()
                new_pair.foiled["regions"].append({"region_number":1, "content": r1})
                r2 = attr.strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            else:
                r1 = earlier.strip()
                new_pair.foiled["regions"].append({"region_number":1, "content": r1})
                r2 = attr.strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            r3 = later.strip()
            new_pair.foiled["regions"].append({"region_number":3, "content": r3})
            full_pairs.append(new_pair)

    return full_pairs


def caption_adj_opposite_combinator(pairs, config):
    """
    Generate foil text by replacing an adjective
    with an attribute from a different context.
    """

    attrs = gu.vg_as_dict(config, "attributes", keys="visgen")
    p = inflect.engine()

    index_path = os.path.join(config["Datasets"]["vg_path"], "w2v_attributes.index")
    model_path = os.path.join(config["Datasets"]["vg_path"], "w2v_attributes.model")
    if os.path.exists(index_path):
        index = AnnoyIndexer()
        index.load(index_path)
        model = Word2Vec.load(model_path)
    else:
        corpus = []
        for img_id in attrs:
            img = attrs[img_id]
            img_attrs = []
            for obj in img['attributes']:
                if not 'attributes' in obj:
                    continue
                for a in obj['attributes']:
                    img_attrs += a.split()
            corpus.append(img_attrs)

        model = Word2Vec(sentences=corpus, vector_size=100, window=20, min_count=1, sg=0)
        model.save(model_path)
        index = AnnoyIndexer(model, 100)
        index.save(index_path)

    full_pairs = []
    empty = 0
    #for pair in tqdm(pairs):
    for pair in pairs:
        orig_attr = pair.info["orig_adj"]
        negative = orig_attr.split()
        negative = [w for w in negative if w in model.wv]
        if negative == []:
            empty += 1
            continue

        foil_attrs = model.wv.most_similar(negative=negative, topn=5, indexer=index)
        for attr, sim in foil_attrs:
            new_pair = copy.deepcopy(pair)
            earlier, later = new_pair.context
            #if "young child" in earlier or "tennis ball" in later:
            #    print(earlier, later, attr)
            if 'start' in new_pair.info and new_pair.info['start'] == True:
                new_pair.foiled["regions"].append({"region_number":1, "content": ""})
                r2 = attr.capitalize().strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            elif new_pair.info['indefinite']:
                article = p.a(attr).split()[0]
                if len(earlier) > 0:
                    r1 = earlier + " " + article#+ p.a(attr).strip()
                else:
                    r1 = article.capitalize()#p.a(attr).strip()
                r1 = r1.strip()
                new_pair.foiled["regions"].append({"region_number":1, "content": r1})
                r2 = attr.lower().strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            else:
                r1 = earlier.strip()
                new_pair.foiled["regions"].append({"region_number":1, "content": r1})
                r2 = attr.strip()
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
        foil_objs = random.choices(foil_objs, k=10)
        for obj in foil_objs:
            new_pair = copy.deepcopy(pair)

            r1 = pair.context[0]
            new_pair.correct["regions"].append({"region_number": 1, "content": r1})
            r2 = pair.info["orig_obj"]
            new_pair.correct["regions"].append({"region_number": 2, "content": r2})

            new_pair.foiled["regions"].append({"region_number": 1, "content": r1})
            if "name" in obj:
                r2 = obj["name"].strip('"')
            else:
                r2 = obj["names"][0].strip('"')
            new_pair.foiled["regions"].append({"region_number": 2, "content": r2})

            new_pair.region_meta = {"1": "context", "2": "object"}
            new_pair.formula = "(2;%foiled%) > (2;%correct%)"
            full_pairs.append(new_pair)

    return full_pairs


def relationship_obj_opposite_combinator(pairs, config):
    """
    Counterpart to relationship_obj_generator.
    """

    objs = gu.vg_as_dict(config, "objects", keys="visgen")

    index_path = os.path.join(config["Datasets"]["vg_path"], "w2v_objects.index")
    model_path = os.path.join(config["Datasets"]["vg_path"], "w2v_objects.model")
    if os.path.exists(index_path):
        index = AnnoyIndexer()
        index.load(index_path)
        model = Word2Vec.load(model_path)
    else:
        corpus = []
        for img_id in objs:
            img = objs[img_id]
            img_objs = []
            for obj in img['objects']:
                if not 'names' in obj:
                    continue
                for name in obj['names']:
                    img_objs += name.lower().split()
            corpus.append(img_objs)

        model = Word2Vec(sentences=corpus, vector_size=100, window=20, min_count=1, sg=0)
        model.save(model_path)
        index = AnnoyIndexer(model, 100)
        index.save(index_path)

    full_pairs = []
    for pair in pairs:
        orig_obj = pair.info["orig_obj"]

        all_objs = objs[pair.orig_img]["objects"]
        negative = []
        for obj in all_objs:
             names = obj["names"]
             for name in names:
                 name = name.lower().split()
                 negative += name
        #negative = orig_obj.split()
        negative = [w for w in negative if w in model.wv]
        if negative == []:
            continue

        foil_objs = model.wv.most_similar(negative=negative, topn=5, indexer=index)
        for obj, sim in foil_objs:
            new_pair = copy.deepcopy(pair)

            r1 = pair.context[0]
            new_pair.correct["regions"].append({"region_number": 1, "content": r1})
            r2 = pair.info["orig_obj"]
            new_pair.correct["regions"].append({"region_number": 2, "content": r2})

            new_pair.foiled["regions"].append({"region_number": 1, "content": r1})
            r2 = obj.strip()
            new_pair.foiled["regions"].append({"region_number": 2, "content": r2})

            new_pair.region_meta = {"1": "context", "2": "object"}
            new_pair.formula = "(2;%foiled%) > (2;%correct%)"
            full_pairs.append(new_pair)

    return full_pairs


def qa_base_combinator(pairs, config):
    """
    Counterpart to qa_base_generator.
    """

    qas = gu.vg_as_dict(config, "question_answers.json", keys="visgen")

    full_pairs = []
    for pair in pairs:
        foil_qas = qas[pair.foil_img]['qas']
        foil_qas = random.choices(foil_qas, k=10)
        for qa in foil_qas:
            full_pair = copy.deepcopy(pair)
            r1 = pair.context
            full_pair.foiled["regions"].append({"region_number":1, "content": r1})
            r2 = qa["answer"]
            full_pair.foiled["regions"].append({"region_number":2, "content": r2})

            full_pairs.append(full_pair)

    return full_pairs


def vg_obj_list_combinator(pairs, config):
    """
    Counterpart to vg_obj_list_generator.
    """

    objs = gu.vg_as_dict(config, "objects", keys="visgen")
    p = inflect.engine()

    full_pairs = []
    for pair in pairs:
        orig_obj = pair.info["corr_obj"]
        article = p.a(orig_obj).split()[0]
        r1 = pair.context + " " + article
        pair.correct["regions"].append({"region_number":1, "content": r1})
        r2 = orig_obj
        pair.correct["regions"].append({"region_number":2, "content": r2})
        pair.correct["regions"].append({"region_number":3, "content": "."})

        pair.region_meta = {"1": "context", "2": "object", "3": "end"}
        pair.formula = "(2;%foiled%) > (2;%correct%)"

        foil_objs = objs[pair.foil_img]["objects"]
        foil_objs = set([o["names"][0] for o in foil_objs])
        #if len(foil_objs) < 10:
        #    foil_objs = random.choices(foil_objs, k=len(foil_objs)-1)
        #else:
        #    foil_objs = random.choices(foil_objs, k=10)

        for foil_obj in foil_objs:
            full_pair = copy.deepcopy(pair)
            article = p.a(foil_obj).split()[0]
            r1 = pair.context + " " + article
            full_pair.foiled["regions"].append({"region_number":1, "content": r1})
            full_pair.foiled["regions"].append({"region_number":2, "content": foil_obj})
            full_pair.foiled["regions"].append({"region_number":3, "content": "."})

            full_pairs.append(full_pair)

    return full_pairs


def vg_obj_list_opposite_combinator(pairs, config):
    """
    Counterpart to vg_obj_list_generator,
    but choose foil word based on word similarities
    in the visual genome objects.
    """

    objs = gu.vg_as_dict(config, "objects", keys="visgen")
    p = inflect.engine()

    index_path = os.path.join(config["Datasets"]["vg_path"], "w2v_objects.index")
    model_path = os.path.join(config["Datasets"]["vg_path"], "w2v_objects.model")
    if os.path.exists(index_path):
        index = AnnoyIndexer()
        index.load(index_path)
        model = Word2Vec.load(model_path)
    else:
        corpus = []
        for img_id in objs:
            img = objs[img_id]
            img_objs = []
            for obj in img['objects']:
                if not 'names' in obj:
                    continue
                for name in obj['names']:
                    img_objs += name.split()
            corpus.append(img_objs)

        model = Word2Vec(sentences=corpus, vector_size=100, window=20, min_count=1, sg=0)
        model.save(model_path)
        index = AnnoyIndexer(model, 100)
        index.save(index_path)

    empty = 0
    full_pairs = []
    for pair in pairs:
        #r1 = pair.context
        orig_obj = pair.info["corr_obj"]
        article = p.a(orig_obj).split()[0]
        r1 = pair.context + " " + article
        pair.correct["regions"].append({"region_number":1, "content": r1})
        pair.correct["regions"].append({"region_number":2, "content": orig_obj})
        pair.correct["regions"].append({"region_number":3, "content": "."})

        pair.region_meta = {"1": "context", "2": "object", "3": "end"}
        pair.formula = "(2;%foiled%) > (2;%correct%)"

        #negative = orig_obj.split()
        negative = []
        all_objs = objs[pair.orig_img]["objects"]
        for obj in all_objs:
             names = obj["names"]
             for name in names:
                 name = name.lower().split()
                 negative += name

        negative = [w for w in negative if w in model.wv]
        if negative == []:
            empty += 1
            continue

        foil_objs = model.wv.most_similar(negative=negative, topn=5, indexer=index)

        for foil_obj, sim in foil_objs:
            full_pair = copy.deepcopy(pair)
            #r1 = pair.context
            #full_pair.foiled["regions"].append({"region_number":1, "content": r1})
            #r2 = p.a(foil_obj) + "."
            #full_pair.foiled["regions"].append({"region_number":2, "content": r2})
            article = p.a(foil_obj).split()[0]
            r1 = pair.context + " " + article
            full_pair.foiled["regions"].append({"region_number":1, "content": r1})
            full_pair.foiled["regions"].append({"region_number":2, "content": foil_obj})
            full_pair.foiled["regions"].append({"region_number":3, "content": "."})

            full_pairs.append(full_pair)

    return full_pairs


def vg_attribute_opposite_combinator(pairs, config):
    """
    Like vg_attribute_combinator,
    but choose foil word based on word similarities
    in the visual genome attributes.
    """

    attrs = gu.vg_as_dict(config, "attributes", keys="visgen")
    for imgid, img in attrs.items():
        if not isinstance(img, dict):
            print(type(img))
    p = inflect.engine()
    new_pairs = []

    index_path = os.path.join(config["Datasets"]["vg_path"], "w2v_attributes.index")
    model_path = os.path.join(config["Datasets"]["vg_path"], "w2v_attributes.model")
    if os.path.exists(index_path):
        index = AnnoyIndexer()
        index.load(index_path)
        model = Word2Vec.load(model_path)
    else:
        corpus = []
        for img_id in attrs:
            img = attrs[img_id]
            img_attrs = []
            for obj in img['attributes']:
                if not 'attributes' in obj:
                    continue
                for a in obj['attributes']:
                    img_attrs += a.split()
            corpus.append(img_attrs)

        model = Word2Vec(sentences=corpus, vector_size=100, window=20, min_count=1, sg=0)
        model.save(model_path)
        index = AnnoyIndexer(model, 100)
        index.save(index_path)

    empty = 0
    for pair in pairs:
        orig_obj = pair.info["orig_object"]
        attr = random.choice(orig_obj["attributes"]).lower()
        #r = pair.context[0] + p.a(attr)
        #r += " " + pair.context[1] + "."
        #pair.correct['regions'].append({"region_number":1, "content": r})
        article = p.a(attr).split()[0]
        earlier = pair.context[0] + " " + article
        earlier = earlier.strip()
        pair.correct['regions'].append({"region_number":1, "content": earlier})
        pair.correct['regions'].append({"region_number":2, "content": attr})
        later = pair.context[1].strip() + "."
        pair.correct['regions'].append({"region_number":3, "content": later})

        negative = []
        for attr in orig_obj["attributes"]:
            negative += attr.lower().split()
        #negative = attr.split()
        negative = [w for w in negative if w in model.wv]
        if negative == []:
            empty += 1
            continue

        foil_attrs = model.wv.most_similar(negative=negative, topn=5, indexer=index)
        for (attr, sim) in foil_attrs:
            attr = attr.strip().lower()
            #r = pair.context[0] + p.a(attr)
            #r += " " + pair.context[1] + "."
            new_pair = copy.deepcopy(pair)
            #new_pair.foiled['regions'].append({"region_number":1, "content": r})

            #new_pair.region_meta = {"1": "sentence"}
            #new_pair.formula = "(1;%foiled%) > (1;%correct%)"
            #new_pairs.append(new_pair)

            article = p.a(attr).split()[0]
            earlier = pair.context[0] + " " + article
            earlier = earlier.strip()
            new_pair.foiled['regions'].append({"region_number":1, "content": earlier})
            new_pair.foiled['regions'].append({"region_number":2, "content": attr})
            new_pair.correct['regions'].append({"region_number":3, "content": later})

            new_pair.region_meta = {"1": "context", "2": "attribute", "3": "object"}
            new_pair.formula = "(2;%foiled%) > (2;%correct%)"

            new_pairs.append(new_pair)

    return new_pairs


def ade_same_object_combinator(pairs, config):
    """
    Generate following the pattern "This is a [category1]. There is an [object]."
    vs. "This is a [category2]. There is an [object]."
    """
    p = inflect.engine()
    nlp = spacy.load("en_core_web_sm")
    pairs_with_text = []

    if len(pairs) > 20000:
        pairs = random.choices(pairs, k=20000)

    #for pair in tqdm(pairs):
    for pair in pairs:
        obj = pair.info["obj"]
        orig_cat = pair.info["orig_cat"]
        r1 = gu.ade_cat2text(orig_cat)[:-1] # strip full stop
        r1 = r1.split()
        earlier = " ".join(r1[:3])
        pair.correct['regions'].append({"region_number":1, "content": earlier})
        cat = " ".join(r1[3:])
        pair.correct['regions'].append({"region_number":2, "content": cat})
        #pair.correct['regions'].append({"region_number":1, "content": r1})

        #if nlp(obj)[0].lemma_ == obj:
        obj_nlp = nlp(obj)[-1]
        if obj_nlp.lemma_ == obj_nlp.text:
        ##if " ".join([o.lemma_ for o in nlp(obj)]) == " ".join(obj.split()):
            singular = True
        else:
            singular = False
        if singular:
            r2 = ". There is " + p.a(obj) + "."
        else:
            r2 = ". There are " + obj + "."
        pair.correct['regions'].append({"region_number":3, "content": r2})

        foil_cat = pair.info["foil_cat"]
        r1 = gu.ade_cat2text(foil_cat)[:-1]
        r1 = r1.split()
        earlier = " ".join(r1[:3])
        pair.foiled['regions'].append({"region_number":1, "content": earlier})
        cat = " ".join(r1[3:])
        pair.foiled['regions'].append({"region_number":2, "content": cat})
        pair.foiled['regions'].append({"region_number":3, "content": r2})
        #pair.foiled['regions'].append({"region_number":1, "content": r1})
        #pair.foiled['regions'].append({"region_number":2, "content": r2})
        pairs_with_text.append(pair)

    return pairs_with_text


def ade_same_category_combinator(pairs, config):
    """
    Generate following the pattern "This is a [category]. There is an [object1]."
    vs. "This is a [category]. There is an [object2]."
    """
    p = inflect.engine()
    nlp = spacy.load("en_core_web_sm")
    pairs_with_text = []

    if len(pairs) > 20000:
        pairs = random.choices(pairs, k=20000)

    for pair in pairs:
        cat = pair.info["category"]
        orig_obj = pair.info["orig_obj"]
        r1 = gu.ade_cat2text(cat)
        #pair.correct['regions'].append({"region_number":1, "content": r1})

        #if nlp(orig_obj)[0].lemma_ == orig_obj:
        obj_nlp = nlp(orig_obj)[-1]
        if obj_nlp.lemma_ == obj_nlp.text:
        #if nlp(orig_obj)[-1].lemma_ == orig_obj.split()[-1]:
            singular = True
        else:
            singular = False
        #if singular:
        #    r2 = "There is " + p.a(orig_obj) + "."
        #else:
        #    r2 = "There are " + orig_obj + "."
        #pair.correct['regions'].append({"region_number":2, "content": r2})
        if singular:
            article = p.a(orig_obj).split()[0]
            #correct_text = "There is " + p.a(orig_obj_name) + "."
            correct_text = "There is " + article
        else:
            #correct_text = "There are " + orig_obj_name + "."
            correct_text = "There are" #+ orig_obj_name + "."
        earlier = r1 + " " + correct_text
        #pair.correct["regions"].append({"region_number":1, "content":pair.context[0]})
        #pair.correct["regions"].append({"region_number":2, "content":correct_text})
        pair.correct["regions"].append({"region_number":1, "content":earlier})
        pair.correct["regions"].append({"region_number":2, "content":orig_obj})
        pair.correct["regions"].append({"region_number":3, "content":"."})

        #pair.foiled['regions'].append({"region_number":1, "content": r1})

        foil_obj = pair.info["foil_obj"]
        #if nlp(foil_obj)[0].lemma_ == foil_obj:
        obj_nlp = nlp(foil_obj)[-1]
        if obj_nlp.lemma_ == obj_nlp.text:
        #if nlp(foil_obj)[-1].lemma_ == foil_obj.split()[-1]:
            singular = True
        else:
            singular = False
        #if singular:
        #    r2 = "There is " + p.a(foil_obj) + "."
        #else:
        #    r2 = "There are " + foil_obj + "."
        if singular:
            article = p.a(foil_obj).split()[0]
            #correct_text = "There is " + p.a(orig_obj_name) + "."
            correct_text = "There is " + article
        else:
            #correct_text = "There are " + orig_obj_name + "."
            correct_text = "There are" #+ orig_obj_name + "."
        earlier = r1 + " " + correct_text
        pair.foiled["regions"].append({"region_number":1, "content":earlier})
        pair.foiled["regions"].append({"region_number":2, "content":foil_obj})
        pair.foiled["regions"].append({"region_number":3, "content":"."})

        #pair.foiled['regions'].append({"region_number":2, "content": r2})
        pairs_with_text.append(pair)

    return pairs_with_text


def caption_pair_combinator(pairs, config):
    """
    Counterpart to caption_pair_generator.
    """

    cxc_path = config["Datasets"]["cxc_path"]
    cap_sim = gu.read_cxc_sentence_similarities(cxc_path)
    coco_dict = gu.coco_as_dict_list(config)

    if "cxc_caption_similarity" in config["Other"]:
        similarity_func = "cxc"
    else:
        similarity_func = "jaccard"

    if "low_sim" in config["Other"]:
        sim_cutoff = 0.2
    else:
        sim_cutoff = 0.4

    new_pairs = []
    for pair in pairs:
        foil_img = pair.foil_img
        context_cap = pair.info["context_cap_obj"]
        if similarity_func == "jaccard":
            #captions = coco_dict[foil_img["img_id"]]
            #captions = coco_dict[foil_img]
            captions = pair.info["captions_foil"]
            jaccs = gu.get_jaccard_similarities(captions, context_cap)
            sims_low = [triple for triple in jaccs if triple[2] < sim_cutoff]
        else:
            #captions = coco_dict[foil_img["img_id"]]
            captions = coco_dict[foil_img]
            sims = gu.get_cxc_cap_similarities(cap_sim, captions, context_cap)
            sims_low = [triple for triple in sims if triple[2] < sim_cutoff]

        context = pair.context[0]
        for (c1, c2, sim) in sims_low:
            new_cap_text = c2["caption"]
            new_pair = copy.deepcopy(pair)
            new_pair.foiled['regions'].append({'region_number':1, 'content': context})
            new_pair.foiled['regions'].append({'region_number':2, 'content': new_cap_text})
            new_pair.info["2nd_cap_object"] = c2
            new_pairs.append(new_pair)

    return new_pairs
