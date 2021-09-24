import json

from context_generators import FoilPair, ade_thereis_generator
from foil_image_selectors import ade_same_env_selector
from combinators import ade_thereis_combinator

ade_path = "/home/johann/Studium/MA/datasets/ADE20k/ADE20K/dataset/ADE20K_2021_17_01/images/ADE/training"

def generate_suite(pairs, save_path):
    """
    Turn pairs produced by the combinator into a syntaxgym suite.

    Parameters
    ----------
    pairs : [FoilPair]
        Duh.
    save_path : str
        Filepath to save the suite to.
    """

    suite = dict()
    suite["meta"] = dict()
    suite["meta"]["name"] = "ade_test" # TODO: naming based on example generation
    suite["meta"]["metric"] = "average" # TODO: check with syntaxgym implementation
    suite["predictions"] = [{"type": "formula", "formula": pairs[0].formula}]
    suite["region_meta"] = pairs[0].region_meta
    suite["items"] = []

    for i, pair in enumerate(pairs):
        item = dict()
        item["item_number"] = i
        item["conditions"] = [pair.correct, pair.foiled]
        suite["items"].append(item)

    with open(save_path, "w") as suite_file:
        json.dump(suite, suite_file)

if __name__ == "__main__":
    pairs = ade_thereis_generator(ade_path, num_examples=10)
    with_selected_imgs = ade_same_env_selector(pairs)
    combined = ade_thereis_combinator(with_selected_imgs)

    generate_suite(combined, "test.json")
