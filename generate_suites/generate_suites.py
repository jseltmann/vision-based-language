from context_generators import FoilPair, ade_thereis_generator
from foil_image_selectors import ade_same_env_selector
from combinators import ade_thereis_combinator

ade_path = "/home/johann/Studium/MA/datasets/ADE20k/ADE20K/dataset/ADE20K_2021_17_01/images/ADE/training"

if __name__ == "__main__":
    pairs = ade_thereis_generator(ade_path, num_examples=10)
    with_selected_imgs = ade_same_env_selector(pairs)
    combined = ade_thereis_combinator(with_selected_imgs)
    for pair in combined:
        print(pair.correct, pair.foiled)
