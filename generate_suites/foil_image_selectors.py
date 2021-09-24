import os
import random

from context_generators import FoilPair

random.seed(0)

def ade_same_env_selector(pairs):
    """
    Select an image from the same ADE20k category as foil image.

    Parameters
    ----------
    pairs : [FoilPair]
        List of foil pairs without foil images.

    Returns
    -------
    pairs_with_foil : [FoilPair]
        Pairs with selected foil image.
    """

    pairs_with_foil = []
    for pair in pairs:
        orig_path = pair.orig_img
        orig_fn = os.path.basename(orig_path)
        category_path = os.path.dirname(orig_path)
        
        imgs = os.listdir(category_path)
        imgs = [i for i in imgs if i.endswith("json")]
        imgs.remove(orig_fn)
        foil_img = random.choice(imgs)
        foil_path = os.path.join(category_path, foil_img)

        pair.foil_img = foil_path
        pairs_with_foil.append(pair)

    return pairs_with_foil
