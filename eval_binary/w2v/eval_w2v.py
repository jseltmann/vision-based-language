import spacy
import pickle
import gensim.downloader
import numpy as np

class W2V_classifier:
    def __init__(self, trained_path):
        with open(trained_path, "rb") as clsf:
            self.classifier = pickle.load(clsf)
        self.wv = gensim.downloader.load('glove-wiki-gigaword-50')
        self.nlp = spacy.load("en_core_web_sm")

    def classify(self, example):
        svs = []
        for sent in example:
            sv = np.zeros(50)
            toks = [t.text.lower() for t in self.nlp.tokenizer(sent)]
            for tok in toks:
                if tok not in self.wv:
                    continue
                sv += self.wv[tok]
            svs.append(sv)
        combinedv = np.append(svs[0], [svs[1]])
        combinedv = np.expand_dims(combinedv, axis=0)
        outp = self.classifier.predict(combinedv)
        return outp
