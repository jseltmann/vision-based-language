"""
Tokenization utilities for a transformers model.
"""

import argparse
import os
import logging
from pathlib import Path
import sys

import torch
import numpy as np

#from transformers import AutoTokenizer
import transformers as tr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def readlines(inputf):
    with inputf as f:
        lines = f.readlines()
    lines = [l.strip('\n') for l in lines]
    return lines

def tokenize_sentence(sentence, tokenizer):
    sent_tokens = tokenizer.tokenize(sentence)
    return sent_tokens

def unkify_sentence(sentence, tokenizer):
    #sent_token_ids = tokenizer.encode(sentence)
    sent_tokens = tokenizer.tokenize(sentence)
    sent_token_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
    unk_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
    return ["1" if idx == unk_id else "0" for idx in sent_token_ids]

def main(args):
    logger.info("Loading tokenizer")
    #tokenizer = AutoTokenizer.from_pretrained(str(args.model_path))
    tokenizer = tr.LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

    logger.info("Reading sentences from %s", args.inputf)
    sentences = readlines(args.inputf)

    f = tokenize_sentence if args.mode == "tokenize" else unkify_sentence
    with args.outputf as of:
        for sentence in sentences:
            of.write(" ".join(f(sentence, tokenizer)) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputf", type=argparse.FileType("r", encoding="utf-8"), help="Input file")
    p.add_argument("-m", "--mode", choices=["tokenize", "unkify"])
    p.add_argument("--model_path", default=None, type=Path, required=True,
                   help="Path to model directory containing checkpoint, vocabulary, config, etc.")
    p.add_argument('--outputf', '-o', type=argparse.FileType("w"), default=sys.stdout,
                   help='output file for generated text')

    main(p.parse_args())
