import json
import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pickle


class SpacyTokenizer():
    def __init__(self):
        nlp = spacy.load("en_core_web_sm")
        self.tokenizer = nlp.tokenizer

    def __call__(self, text):
        doc = self.tokenizer(text)
        return [t.text for t in doc]

def train_bow():
    with open("finetune.json") as dataf:
        data = json.loads(dataf.read())

    docs = data["true"] + data["false"]

    #SpacyTokenizer.__module__ = "train_bow"
    tok = SpacyTokenizer()
    pipeline = Pipeline([("tfidf", TfidfVectorizer(tokenizer=tok)),
                          ("classifier", LinearSVC())])
    pipeline.fit(docs, [1] * len(data["true"]) + [0] * len(data["false"]))

    with open("classifier.pkl", "wb") as clsf:
        pickle.dump(pipeline, clsf)

if __name__=="__main__":
    train_bow()
