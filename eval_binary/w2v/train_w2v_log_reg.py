import json
import spacy
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle
import gensim.downloader
import numpy as np
import random
import docker
import os
import shutil
import csv

random.seed(0)

def train_w2v():
    gc_path = "/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv"
    with open(gc_path, newline='') as gcf:
        not_comment = lambda line: line[0]!='#'
        reader = csv.reader(filter(not_comment, gcf), delimiter=",")
        nlp = spacy.load("en_core_web_sm")
        wv = gensim.downloader.load('glove-wiki-gigaword-50')
        for i, row in enumerate(reader):
            for add in ["", "_cap"]:
                if i == 0:
                    continue
                suite_name = row[4] + add
                suites_dir = os.path.join("/home/jseltmann/data/suites_coco_val", row[3])
                suite_path = os.path.join(suites_dir, suite_name + "_train.json")

                tag = suite_name
                print(tag)
                full_name = "w2v-" + tag
                if not os.path.exists(suite_path):
                    continue

                with open(suite_path) as dataf:
                    data = json.loads(dataf.read())
                if len(data) == 0:
                    continue

                vectors = []
                labels = []
                for pair in data:
                    label = pair[1]
                    svs = []
                    for sent in pair[0]:
                        sv = np.zeros(50)
                        toks = [t.text.lower() for t in nlp.tokenizer(sent)]
                        for tok in toks:
                            if tok not in wv:
                                continue
                            sv += wv[tok]
                        svs.append(sv)
                    combinedv = np.append(svs[0], [svs[1]])
                    vectors.append(combinedv)
                    labels.append(label)

                classifier = LogisticRegression(max_iter=10000, solver='sag')
                try:
                    classifier.fit(vectors, labels)
                except Exception as e:
                    print(e)

                with open("log_reg_addition/"+full_name+".pkl", "wb") as clsf:
                    pickle.dump(classifier, clsf)


if __name__=="__main__":
    train_w2v()
