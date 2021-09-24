import json
import configparser
import sys

import context_generators as cg
import foil_image_selectors as fis
import combinators as com

def generate_suite(pairs, save_path, suite_name, functions):
    """
    Turn pairs produced by the combinator into a syntaxgym suite.

    Parameters
    ----------
    pairs : [FoilPair]
        Duh.
    save_path : str
        Filepath to save the suite to.
    suite_name : str
        Name of suite.
    functions : (str, str, str)
        Names of (generator, selector, combinator) functions used to create suite.
    """

    suite = dict()
    suite["meta"] = dict()
    suite["meta"]["name"] = suite_name
    suite["meta"]["metric"] = "average" # TODO: check with syntaxgym implementation
    suite["meta"]["comments"] = dict()
    suite["meta"]["comments"]["functions"] = functions
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

    if len(sys.argv) != 2:
        print("Using default config file.")
        config_path = "generation.config"
    else:
        config_path = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(config_path)

    generator_str = config["Functions"]["generator"] # read name of generator function from config file
    generator = getattr(cg, generator_str)
    selector_str = config["Functions"]["selector"]
    selector = getattr(fis, selector_str)
    combinator_str = config["Functions"]["combinator"]
    combinator = getattr(com, combinator_str)

    pairs = generator(config_path)
    with_foil_imgs = selector(pairs)
    combined = combinator(with_foil_imgs)

    save_path = config["General"]["save_path"]
    suite_name = config["General"]["suite_name"]

    generate_suite(combined, save_path, suite_name, 
            (generator_str, selector_str, combinator_str))
