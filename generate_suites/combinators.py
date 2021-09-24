import os
import random
import json
import inflect

from context_generators import FoilPair

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
        pair.correct["regions"].append({"region_number":1, "content":pair.context})
        pair.correct["regions"].append({"region_number":2, "content":correct_text})

        foil_annot = json.load(open(pair.foil_img))['annotation']
        foil_obj = random.choice(foil_annot['object'])
        foil_obj_name = foil_obj['raw_name']
        foiled_text = "There is " + p.a(foil_obj_name) + "."
        pair.foiled["regions"].append({"region_number":1, "content":pair.context})
        pair.foiled["regions"].append({"region_number":2, "content":foiled_text})

        pair.region_meta = {"1": "context", "2": "thereis"}
        pair.formula = "(2;%foiled%) > (2;%correct%)"

        pairs_with_texts.append(pair)

    return pairs_with_texts
