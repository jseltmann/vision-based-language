import json
import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
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


def train_bow():
    gc_path = "/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv"
    with open(gc_path, newline='') as gcf:
        not_comment = lambda line: line[0]!='#'
        reader = csv.reader(filter(not_comment, gcf), delimiter=",")
        nlp = spacy.load("en_core_web_sm")
        wv = gensim.downloader.load('glove-wiki-gigaword-50')
        for i, row in enumerate(reader):
            for add in ["", "_cap"]:
                suite_name = row[4] + add
                if not "dissim" in suite_name:
                    continue
                if i == 0:
                    continue
                suites_dir = os.path.join("/home/jseltmann/data/suites_coco_val", row[3])
                suite_path = os.path.join(suites_dir, suite_name + "_train.json")

                tag = suite_name
                print(tag)
                full_name = "w2v-" + tag
                if os.path.exists(full_name+".pkl"):
                    continue
                if not os.path.exists(suite_path):
                    continue

                with_labels = []
                with open(suite_path) as dataf:
                    data = json.loads(dataf.read())
                    for (text1, text2), label in data:
                        if label == 0:
                            with_labels.append((0,text1))
                            with_labels.append((1,text2))
                        else:
                            with_labels.append((1,text1))
                            with_labels.append((0,text2))
                    #docs = data["true"] + data["false"]
                if len(with_labels) == 0:
                    continue

                random.shuffle(with_labels)

                docvs = []
                labels = []
                for label, doc in with_labels:
                    docv = np.zeros(50)
                    toks = [t.text.lower() for t in nlp.tokenizer(doc)]
                    for tok in toks:
                        if tok not in wv:
                            continue
                        docv += wv[tok]
                    docvs.append(docv)
                    labels.append(label)

                # use 0 for true since these values are used by syntaxgym as losses
                # meaning lower is better
                classifier = LogisticRegression(max_iter=10000)
                classifier.fit(docvs, labels)

                with open("classifier.pkl", "wb") as clsf:
                    pickle.dump((wv, classifier), clsf)

                prev_path = "/home/jseltmann/project/vision-based-language/images/"
                cp_path = os.path.join(prev_path, full_name+".pkl")
                shutil.copy("classifier.pkl", cp_path)

                docker_client = docker.from_env()
                try:
                    docker_client.images.remove(image="w2v:"+tag)
                except Exception as e:
                    print(e)
                with open("Dockerfile") as df:
                    docker_client.images.build(path=".", tag="w2v:"+tag, rm=True)

                reg_entry = {"ref_url": "https://me.com/my_lm",
                        "image": {
                          "maintainer": "jseltmann@uni-potsdam.de",
                          "version": "9e86105567472f4dab57944b8e6fa6059d0f61fa",
                          "checksum": "d6916417b16ee864f76c8de67a43db6d6309badd",
                          "datetime": "2021-10-16T18:19:45.139122181Z",
                          "supported_features": {
                            "tokenize": True,
                            "unkify": True,
                            "get_surprisals": True,
                            "get_predictions": False,
                            "mount_checkpoint": False
                          },
                          "gpu": {
                            "required": False,
                            "supported": False
                          },
                          "name": "w2v",
                          "tag": tag,
                          "size": 2438041842
                        },
                        "tokenizer": {
                          "type": "word",
                          "cased": True,
                          "sentinel_position": "initial",
                          "sentinel_pattern": "\u0120"
                        },
                        "shortname": full_name }

                reg_path = "/home/jseltmann/project/vision-based-language/images/registry.json"
                if os.path.exists(reg_path):
                    with open(reg_path) as rf:
                        registry = json.load(rf)
                else:
                    registry = dict()
                registry[full_name] = reg_entry
                with open(reg_path, "w") as rf:
                    json.dump(registry, rf)


if __name__=="__main__":
    train_bow()
