import os
import random
import json
import inflect
import copy

import generation_utils as gu

random.seed(0)

def ade_thereis_combinator(pairs):
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

    pairs_with_texts = []
    p = inflect.engine()

    for pair in pairs:
        orig_annot = json.load(open(pair.orig_img))['annotation']
        orig_obj = random.choice(orig_annot['object'])
        orig_obj_name = orig_obj['raw_name']
        correct_text = "There is " + p.a(orig_obj_name) + "."
        pair.correct["regions"].append({"region_number":1, "content":pair.context[0]})
        pair.correct["regions"].append({"region_number":2, "content":correct_text})

        foil_annot = json.load(open(pair.foil_img))['annotation']
        foil_obj = random.choice(foil_annot['object'])
        foil_obj_name = foil_obj['raw_name']
        foiled_text = "There is " + p.a(foil_obj_name) + "."
        pair.foiled["regions"].append({"region_number":1, "content":pair.context[0]})
        pair.foiled["regions"].append({"region_number":2, "content":foiled_text})

        pair.region_meta = {"1": "context", "2": "thereis"}
        pair.formula = "(2;%foiled%) > (2;%correct%)"

        pairs_with_texts.append(pair)

    return pairs_with_texts

def caption_adj_combinator(pairs, config):
    """
    Generate foil text by replacing an adjective
    with an attribute froma different context.
    """

    #vg_path = config["Datasets"]["vg_path"]
    #with open(os.path.join(vg_path, "attributes.json")) as vgf:
    #    attr_data = json.loads(vgf.read())

    #mscoco_path = config["Datasets"]["mscoco_path"]
    #coco2vg = gu.get_vg_image_id(vg_path, mscoco_path, reverse=True)
    attrs = gu.attrs_as_dict(config)
    p = inflect.engine()

    full_pairs = []
    for pair in pairs:
        # TODO: make this more efficient?
        #vg_foil_id = coco2vg[pair.foil_img]
        #img_attrs = [d for d in attr_data if d['image_id'] == vg_foil_id]
        #try:
        #    img_attr = img_attrs[0] # works bc there is only one entry for each image
        #except Exception as e:
        #    print(pair.foil_img)
        #    raise(e)
        img_attrs = attrs[pair.foil_img]['attributes']
        has_attrs = lambda o: 'attributes' in o.keys() and o['attributes'] != []
        objs = [obj for obj in img_attrs if has_attrs(obj)]
        for obj in objs:
            new_pair = copy.deepcopy(pair)
            attr = random.choice(obj['attributes'])

            earlier, later = new_pair.context
            if 'start' in new_pair.info and new_pair.info['start'] == True:
                new_pair.foiled["regions"].append({"region_number":1, "content": ""})
                r2 = attr.capitalize().strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            elif new_pair.info['indefinite']:
                r1 = earlier + " " + p.a(attr).strip()
                new_pair.foiled["regions"].append({"region_number":1, "content": r1})
                r2 = attr.strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            else:
                r1 = earlier.strip()
                new_pair.foiled["regions"].append({"region_number":1, "content": r1})
                r2 = earlier.strip()
                new_pair.foiled["regions"].append({"region_number":2, "content": r2})
            r3 = later.strip()
            new_pair.foiled["regions"].append({"region_number":3, "content": r3})
            full_pairs.append(new_pair)

    return full_pairs
