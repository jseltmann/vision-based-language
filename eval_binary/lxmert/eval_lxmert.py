import spacy
import pickle
import gensim.downloader
import numpy as np
import transformers as tr
import torch

class LXMERT_classifier:
    def __init__(self, trained_path):
        with open(trained_path, "rb") as clsf:
            self.classifier = pickle.load(clsf)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.tok = tr.LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        self.lxmert = tr.LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
        self.lxmert.to(self.device)
        self.vfeats = torch.zeros(1,5,2048, dtype=torch.float, device=self.device)
        self.vpos = torch.zeros(1,5,4, dtype=torch.float, device=self.device)

    def classify(self, example):
        svs = []
        for sent in example:
            inputs = self.tok(sent, return_tensors='pt')
            inputs.to(self.device)
            outputs = self.lxmert(**inputs, visual_feats=self.vfeats, visual_pos=self.vpos)
            outputs = outputs.pooled_output
            outputs = np.array(outputs.cpu().detach())
            svs.append(outputs)
        combinedv = np.append(svs[0], [svs[1]])
        combinedv = np.expand_dims(combinedv, axis=0)
        outp = self.classifier.predict(combinedv)
        return outp
