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
from nltk.stem import WordNetLemmatizer
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

        objects = orig_annot['object']
        objects = [o['raw_name'] for o in objects]
        orig_obj = random.choice(objects)
        orig_obj_name = gu.remove_accents(orig_obj)
        sn = p.singular_noun(orig_obj_name)
        if not sn or orig_obj_name == sn: # singular
            singular = True
        else:
            singular = False
        if singular:
            article = p.a(orig_obj_name).split()[0]
            correct_text = "There is " + article
        else:
            correct_text = "There are" #+ orig_obj_name + "."
        earlier = pair.context[0] + " " + correct_text
        pair.correct["regions"].append({"region_number":1, "content":earlier})
        pair.correct["regions"].append({"region_number":2, "content":orig_obj_name})
        pair.correct["regions"].append({"region_number":3, "content":"."})

        json_path = gu.get_ade_json_path(pair.foil_img, data_path, index)
        with codecs.open(json_path, "r", "ISO-8859-1") as jfile:
            foil_annot = json.load(jfile)['annotation']
        foil_objs = [o["raw_name"] for o in foil_annot['object']]
        foil_objs = [o for o in foil_objs if o != orig_obj]
        if len(foil_objs) < 1:
            continue
        foil_obj = random.choice(foil_objs)
        foil_obj_name = gu.remove_accents(foil_obj)
        sn = p.singular_noun(foil_obj_name)
        if not sn or foil_obj_name == sn: # singular
            singular = True
        else:
            singular = False

        if singular:
            article = p.a(foil_obj_name).split()[0]
            correct_text = "There is " + article
        else:
            correct_text = "There are" 
        earlier = pair.context[0] + " " + correct_text
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
        attr = " ".join(attr.split()) # remove multiple spaces in attribute
        attr = "".join(attr.split('"'))
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
            attr = random.choice(obj['attributes'])
            attr = " ".join(attr.split()) # remove multiple spaces in attribute
            attr = "".join(attr.split('"')) # remove multiple spaces in attribute

            new_pair = copy.deepcopy(pair)
            article = p.a(attr).split()[0]
            earlier = pair.context[0] + " " + article
            earlier = earlier.strip()
            new_pair.foiled['regions'].append({"region_number":1, "content": earlier})
            new_pair.foiled['regions'].append({"region_number":2, "content": attr})
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
    for pair in pairs:
        orig_attr = pair.info["orig_adj"]
        negative = orig_attr.split()
        negative = [w for w in negative if w in model.wv]
        if negative == []:
            empty += 1
            continue

        foil_attrs = model.wv.most_similar(negative=negative, topn=5, indexer=index)
        for attr, sim in foil_attrs:
            attr = "".join(attr.split('"'))
            attr = "".join(attr.split())
            new_pair = copy.deepcopy(pair)
            earlier, later = new_pair.context
            if 'start' in new_pair.info and new_pair.info['start'] == True:
                new_pair.foiled["regions"].append({"region_number":1, "content": ""})
                r2 = attr.lower().capitalize().strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            elif new_pair.info['indefinite']:
                article = p.a(attr).split()[0]
                if len(earlier) > 0:
                    r1 = earlier + " " + article
                else:
                    r1 = article.capitalize()
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
        negative = [w for w in negative if w in model.wv]
        if negative == []:
            continue

        foil_objs = model.wv.most_similar(negative=negative, topn=5, indexer=index)
        for obj, sim in foil_objs:
            obj = "".join(obj.split('"'))
            obj = "".join(obj.split("'"))
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
            r2 = "".join(qa["answer"].split('"'))
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
        all_objs = pair.info["chosen_objs"]
        sn = p.singular_noun(orig_obj)
        if not sn or orig_obj == sn: # singular
            article = p.a(orig_obj).split()[0]
            r1 = pair.context + " " + article
        else:
            r1 = pair.context
        pair.correct["regions"].append({"region_number":1, "content": r1})
        r2 = orig_obj
        pair.correct["regions"].append({"region_number":2, "content": r2})
        pair.correct["regions"].append({"region_number":3, "content": "."})

        pair.region_meta = {"1": "context", "2": "object", "3": "end"}
        pair.formula = "(2;%foiled%) > (2;%correct%)"

        foil_objs = objs[pair.foil_img]["objects"]
        foil_objs = set([o["names"][0] for o in foil_objs])
        foil_objs = [o for o in foil_objs if not o in all_objs]

        for foil_obj in foil_objs:
            foil_obj = foil_obj.strip('"')
            full_pair = copy.deepcopy(pair)
            sn = p.singular_noun(foil_obj)
            if not sn or foil_obj == sn: # singular
                article = p.a(foil_obj).split()[0]
                r1 = pair.context + " " + article
            else:
                r1 = pair.context
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
        orig_obj = pair.info["corr_obj"]
        orig_obj = "".join(orig_obj.split('"'))
        sn = p.singular_noun(orig_obj)
        if not sn or orig_obj == sn: # singular
            article = p.a(orig_obj).split()[0]
            r1 = pair.context + " " + article
        else:
            r1 = pair.context
        pair.correct["regions"].append({"region_number":1, "content": r1})
        pair.correct["regions"].append({"region_number":2, "content": orig_obj})
        pair.correct["regions"].append({"region_number":3, "content": "."})

        pair.region_meta = {"1": "context", "2": "object", "3": "end"}
        pair.formula = "(2;%foiled%) > (2;%correct%)"

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
            foil_obj = "".join(foil_obj.split('"'))
            sn = p.singular_noun(foil_obj)
            if not sn or foil_obj == sn: # singular
                article = p.a(foil_obj).split()[0]
                r1 = pair.context + " " + article
            else:
                r1 = pair.context
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
        attr = random.choice(orig_obj["attributes"]).lower().strip()
        attr = attr.strip("\x00")
        attr = "".join(attr.split('"'))
        attr = "".join(attr.split())
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
        negative = [w for w in negative if w in model.wv]
        if negative == []:
            empty += 1
            continue

        foil_attrs = model.wv.most_similar(negative=negative, topn=5, indexer=index)
        for (attr, sim) in foil_attrs:
            attr = attr.strip().lower()
            attr = attr.strip("\x00")
            attr = "".join(attr.split('"'))
            attr = "".join(attr.split())
            new_pair = copy.deepcopy(pair)

            article = p.a(attr).split()[0]
            earlier = pair.context[0] + " " + article
            earlier = earlier.strip()
            new_pair.foiled['regions'].append({"region_number":1, "content": earlier})
            new_pair.foiled['regions'].append({"region_number":2, "content": attr})
            new_pair.foiled['regions'].append({"region_number":3, "content": later})

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

    for pair in pairs:
        new_pair = copy.deepcopy(pair)
        obj = new_pair.info["obj"]
        orig_cat = new_pair.info["orig_cat"]
        r1 = gu.ade_cat2text(orig_cat)[:-1] # strip full stop
        r1 = r1.split()
        earlier = " ".join(r1[:3])
        new_pair.correct['regions'].append({"region_number":1, "content": earlier})
        cat = " ".join(r1[3:])
        new_pair.correct['regions'].append({"region_number":2, "content": cat})

        sn = p.singular_noun(obj)
        if not sn or obj == sn: # singular
            singular = True
        else:
            singular = False
        if singular:
            r2 = ". There is " + p.a(obj) + "."
        else:
            r2 = ". There are " + obj + "."
        new_pair.correct['regions'].append({"region_number":3, "content": r2})

        foil_cat = new_pair.info["foil_cat"]
        r1 = gu.ade_cat2text(foil_cat)[:-1]
        r1 = r1.split()
        earlier = " ".join(r1[:3])
        new_pair.foiled['regions'].append({"region_number":1, "content": earlier})
        foil_cat = " ".join(r1[3:])
        new_pair.foiled['regions'].append({"region_number":2, "content": foil_cat})
        new_pair.foiled['regions'].append({"region_number":3, "content": r2})

        new_pair.region_meta = {"1": "before", "2": "category", "3": "thereis"}
        new_pair.formula = "(2;%foiled%) > (2;%correct%)"
        pairs_with_text.append(new_pair)

    return pairs_with_text


def ade_object_part_paper_combinator(pairs, config):
    """
    Generate following the pattern "This is an [object]. The [part1] is [broken|dirty|missing|ugly|beautiful|particularly large/small]."
    vs. "This is an [object]. The [part2] is [...]."
    """
    p = inflect.engine()
    nlp = spacy.load("en_core_web_sm")
    wnl = WordNetLemmatizer()
    pairs_with_text = []

    if len(pairs) > 20000:
        pairs = random.choices(pairs, k=20000)

    for pair in pairs:
        obj = pair.info["object"].split(",")[0] # use the first name in the list
        #r1 = gu.ade_cat2text(cat)
        r1 = "This is " + p.a(obj) + "."
        pair.context = (r1,)

        #sn = p.singular_noun(orig_obj)
        orig_part = pair.info["orig_part"].split(",")[0].strip()
        orig_part = gu.remove_accents(orig_part)
        correct_text = "The " + orig_part
        #if not sn or orig_part == sn: # singular
        lemma = wnl.lemmatize(orig_part)
        if lemma == orig_part: # singular
            correct_text += " is "
        else:
            correct_text += " are "

        foil_part = pair.info["foil_part"].split(",")[0].strip()
        foil_part = gu.remove_accents(foil_part)
        #sn = p.singular_noun(foil_part)
        lemma = wnl.lemmatize(foil_part, 'n')
        foiled_text = "The " + foil_part
        #if not sn or foil_part == sn: # singular
        if lemma == foil_part: # singular
            foiled_text += " is "
        else:
            foiled_text += " are "

        adjectives = ["broken","dirty","missing","ugly", "beautiful", "particularly large", "particularly small"]
        adjective = random.choice(adjectives)

        new_pair = copy.deepcopy(pair)
        #correct_text += adjective + "."
        text = r1 + " " + correct_text + adjective + "."
        new_pair.correct["regions"].append({"region_number":1, "content":text})
        #new_pair.correct["regions"].append({"region_number":2, "content":orig_obj})
        #new_pair.correct["regions"].append({"region_number":3, "content":"."})

        #foiled_text += adjective + "."
        text = r1 + " " + foiled_text + adjective + "."
        new_pair.foiled["regions"].append({"region_number":1, "content":text})
        #new_pair.foiled["regions"].append({"region_number":2, "content":foil_obj})
        #new_pair.foiled["regions"].append({"region_number":3, "content":"."})

        new_pair.region_meta = {"1": "text"}#, "2":"object", "3": "end"}
        new_pair.formula = "(1;%foiled%) > (1;%correct%)"
        pairs_with_text.append(new_pair)

    return pairs_with_text


def ade_same_category_paper_combinator(pairs, config):
    """
    Generate following the pattern "This is a [category]. The [object1] is [broken|dirty|missing|ugly|beautiful|particularly large/small]."
    vs. "This is a [category]. The [object2] is [...]."
    """
    p = inflect.engine()
    nlp = spacy.load("en_core_web_sm")
    wnl = WordNetLemmatizer()
    pairs_with_text = []

    if len(pairs) > 20000:
        pairs = random.choices(pairs, k=20000)

    for pair in pairs:
        cat = pair.info["category"]
        orig_obj = pair.info["orig_obj"].strip()
        orig_obj = gu.remove_accents(orig_obj)
        r1 = gu.ade_cat2text(cat)
        pair.context = (r1,)

        #sn = p.singular_noun(orig_obj)
        correct_text = "The " + orig_obj
        #if not sn or orig_obj == sn: # singular
        lemma = wnl.lemmatize(orig_obj)
        if lemma == orig_obj: # singular
            correct_text += " is "
        else:
            correct_text += " are "

        foil_obj = pair.info["foil_obj"].strip()
        foil_obj = gu.remove_accents(foil_obj)
        #sn = p.singular_noun(foil_obj)
        lemma = wnl.lemmatize(foil_obj, 'n')
        foiled_text = "The " + foil_obj
        #if not sn or foil_obj == sn: # singular
        if lemma == foil_obj: # singular
            foiled_text += " is "
        else:
            foiled_text += " are "

        adjectives = ["broken","dirty","missing","ugly", "beautiful", "particularly large", "particularly small"]
        adjective = random.choice(adjectives)

        new_pair = copy.deepcopy(pair)
        #correct_text += adjective + "."
        text = r1 + " " + correct_text + adjective + "."
        new_pair.correct["regions"].append({"region_number":1, "content":text})
        #new_pair.correct["regions"].append({"region_number":2, "content":orig_obj})
        #new_pair.correct["regions"].append({"region_number":3, "content":"."})

        #foiled_text += adjective + "."
        text = r1 + " " + foiled_text + adjective + "."
        new_pair.foiled["regions"].append({"region_number":1, "content":text})
        #new_pair.foiled["regions"].append({"region_number":2, "content":foil_obj})
        #new_pair.foiled["regions"].append({"region_number":3, "content":"."})

        new_pair.region_meta = {"1": "text"}#, "2":"object", "3": "end"}
        new_pair.formula = "(1;%foiled%) > (1;%correct%)"
        pairs_with_text.append(new_pair)

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
        new_pair = copy.deepcopy(pair)
        cat = new_pair.info["category"]
        orig_obj = new_pair.info["orig_obj"].strip()
        orig_obj = gu.remove_accents(orig_obj)
        r1 = gu.ade_cat2text(cat)

        sn = p.singular_noun(orig_obj)
        if not sn or orig_obj == sn: # singular
            singular = True
        else:
            singular = False
        if singular:
            article = p.a(orig_obj).split()[0]
            correct_text = "There is " + article
        else:
            correct_text = "There are"
        earlier = r1 + " " + correct_text
        new_pair.correct["regions"].append({"region_number":1, "content":earlier})
        new_pair.correct["regions"].append({"region_number":2, "content":orig_obj})
        new_pair.correct["regions"].append({"region_number":3, "content":"."})

        foil_obj = new_pair.info["foil_obj"].strip()
        foil_obj = gu.remove_accents(foil_obj)
        sn = p.singular_noun(foil_obj)
        if not sn or foil_obj == sn: # singular
            singular = True
        else:
            singular = False
        if singular:
            article = p.a(foil_obj).split()[0]
            correct_text = "There is " + article
        else:
            correct_text = "There are"
        earlier = r1 + " " + correct_text
        new_pair.foiled["regions"].append({"region_number":1, "content":earlier})
        new_pair.foiled["regions"].append({"region_number":2, "content":foil_obj})
        new_pair.foiled["regions"].append({"region_number":3, "content":"."})

        new_pair.region_meta = {"1": "context", "2":"object", "3": "end"}
        new_pair.formula = "(2;%foiled%) > (2;%correct%)"
        pairs_with_text.append(new_pair)

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
        if "low_sim" in config["Other"]:
            sim_cutoff = 1
        else:
            sim_cutoff = 2
    else:
        similarity_func = "jaccard"

        if "low_sim" in config["Other"]:
            sim_cutoff = 0.2
        else:
            sim_cutoff = 0.4

    coco_dict = gu.tokenize_caps(coco_dict)

    new_pairs = []
    for i, pair in enumerate(pairs):
        foil_img = pair.foil_img
        context_cap = pair.info["context_cap_obj"]
        captions = coco_dict[foil_img]
        second_cap_orig = pair.info["2nd_cap_obj"]
        captions = [c for c in captions if c is not second_cap_orig]
        if similarity_func == "jaccard":
            jaccs = gu.get_jaccard_similarities(captions, context_cap)
            sims_low = [triple for triple in jaccs if triple[2] < sim_cutoff]
        else:
            sims = gu.get_cxc_cap_similarities(cap_sim, captions, context_cap)
            sims_low = [triple for triple in sims if triple[2] < sim_cutoff]

        context = pair.context[0]
        for (c1, c2, sim) in sims_low:
            new_cap_text = c2["caption"]
            new_cap_text = new_cap_text.strip()
            new_cap_text = "".join(new_cap_text.split('"'))
            new_cap_text = "".join(new_cap_text.split('\''))
            new_cap_text = new_cap_text.split()
            new_cap_text = [t for t in new_cap_text if len(t)>0]
            new_cap_text = " ".join(new_cap_text)
            new_pair = copy.deepcopy(pair)
            new_pair.foiled['regions'].append({'region_number':1, 'content': context})
            new_pair.foiled['regions'].append({'region_number':2, 'content': new_cap_text})
            new_pair.info["2nd_cap_object"] = c2
            new_pairs.append(new_pair)

    return new_pairs
