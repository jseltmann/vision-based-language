import spacy
import pickle
import gensim.downloader
import numpy as np
import transformers as tr
import torch

class BERT_log_reg_classifier:
    def __init__(self, trained_path):
        with open(trained_path, "rb") as clsf:
            self.classifier = pickle.load(clsf)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.tok = tr.BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = tr.BertModel.from_pretrained('bert-base-uncased')
        self.bert.to(self.device)

    def classify(self, example):
        svs = []
        for sent in example:
            inputs = self.tok(sent, return_tensors='pt')
            inputs.to(self.device)
            outputs = self.bert(**inputs).pooler_output
            outputs = np.array(outputs.cpu().detach())
            svs.append(outputs)
        combinedv = np.append(svs[0], [svs[1]])
        combinedv = np.expand_dims(combinedv, axis=0)
        outp = self.classifier.predict(combinedv)
        return outp
