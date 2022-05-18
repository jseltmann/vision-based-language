import json
import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import pickle
import gensim.downloader
import numpy as np
import random
import docker
import os
import shutil
import csv
import transformers as tr
from tqdm import tqdm
import torch
import numpy as np

random.seed(0)

def train_lxmert():
    device = "cuda" if torch.cuda.is_available() else "gpu"
    gc_path = "/home/jseltmann/project/vision-based-language/generate_suites/generation_combinations_pairwise.csv"
    with open(gc_path, newline='') as gcf:
        not_comment = lambda line: line[0]!='#'
        reader = csv.reader(filter(not_comment, gcf), delimiter=",")
        tok = tr.LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        lxmert = tr.LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
        lxmert.to(device)
        for i, row in enumerate(reader):
            for add in ["", "_cap"]:
                if i == 0:
                    continue
                suite_name = row[4] + add
                suites_dir = os.path.join("/home/jseltmann/data/suites_coco_val", row[3])
                suite_path = os.path.join(suites_dir, suite_name + "_train.json")

                tag = suite_name
                print(tag)
                full_name = "lxmert-" + tag
                if os.path.exists("log_reg_addition/"+full_name+".pkl"):
                    continue
                if not os.path.exists(suite_path):
                    continue

                with open(suite_path) as dataf:
                    data = json.loads(dataf.read())
                if len(data) == 0:
                    continue

                vectors = []
                labels = []
                vfeats = torch.zeros(1,5,2048, dtype=torch.float, device=device)
                vpos = torch.zeros(1,5,4, dtype=torch.float, device=device)
                for pair in tqdm(data):
                    label = pair[1]
                    svs = []
                    for sent in pair[0]:
                        inputs = tok(sent, return_tensors="pt")
                        inputs.to(device)
                        outputs = lxmert(**inputs, visual_feats=vfeats, visual_pos=vpos, output_hidden_states=True)
                        outputs = outputs.language_hidden_states[-1]
                        outputs = np.array(outputs.cpu().detach())
                        outputs = np.sum(outputs, axis=1)
                        svs.append(outputs)
                    combinedv = np.append(svs[0], [svs[1]])
                    vectors.append(combinedv)
                    labels.append(label)

                classifier = LogisticRegression(max_iter=10000, solver='sag')
                try:
                    classifier.fit(vectors, labels)
                except Exception as e:
                    print(e)
                    continue

                with open("log_reg_addition/"+full_name+".pkl", "wb") as clsf:
                    pickle.dump(classifier, clsf)


if __name__=="__main__":
    train_lxmert()
