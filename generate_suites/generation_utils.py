import os
import random


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
