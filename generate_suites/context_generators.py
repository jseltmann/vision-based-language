import os
import random
import json
import nltk
import inflect

random.seed(0)

class FoilPair:
    """
    A pair of two text snippets, where in one of the snippets, 
    some part of the text was replaced by foil text.

    Attributes
    ----------
    context : str
        Basic context produced by context_generator.
    orig_img : str
        Path to annotations of original image.
    foil_img : str
        Path to annotations of image chosen to select foil word.
    correct : dict
        Condition in syntaxgym suite format containing the correct text.
    foiled : dict
        Condition in syntaxgym suite format containing the foiled text.
    region_meta : dict
        Region names in the format required by the sntaxgym json representation.
    formula : str
        Formula for syntaxgym to determine result for pair.
    """
    def __init__(self, context, orig_img):
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

def ade_thereis_generator(data_path, num_examples):
    """
    Generate context based on ADE20k dataset.
    Each examples gives the scene name and an object contained in it,
    e.g. "This is a street. There is an umbrella.".

    Parameters
    ----------
    data_path: str
        Path to ADE20k dataset.
    num_examples: int
        Number of examples to generate.

    Return
    ------
    pairs : [FoilPair]
        List of FoilPairs with the foil examples not yet set.
    """

    p = inflect.engine()

    environments = os.listdir(data_path)
    env_paths = [os.path.join(data_path, e) for e in environments]
    specific_env_paths = []
    for ep in env_paths:
        specific_envs = os.listdir(ep)
        specific_env_paths += [os.path.join(ep, spe) for spe in specific_envs]
    img_info_paths = []
    for sep in specific_env_paths:
        img_info_paths += [os.path.join(sep, ip) for ip in os.listdir(sep) if ip.endswith("json")]
    #TODO: move above to utils

    pairs = []

    base_img_paths = random.choices(img_info_paths, k=num_examples)
    for imgp in base_img_paths:
        annot = json.load(open(imgp))['annotation']
        scene = random.choice(annot['scene'])
        synset = nltk.corpus.wordnet.synsets(scene)[0]
        if synset.pos() == "n":
            context = "This is " + p.a(scene) + "."
        else:
            context = "This is " + p.a(scene) + " place."

        #obj = random.choice(annot['object'])
        #obj_name = obj['raw_name']
        #correct = context + p.a(obj_name) + "."

        pair = FoilPair(context, imgp)
        pairs.append(pair)

    return pairs
