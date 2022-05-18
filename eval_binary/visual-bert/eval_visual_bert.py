import spacy
import pickle
import gensim.downloader
import numpy as np
import transformers as tr
import torch

class Visual_BERT_classifier:
    def __init__(self, trained_path):
        with open(trained_path, "rb") as clsf:
            self.classifier = pickle.load(clsf)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.tok = tr.BertTokenizer.from_pretrained('bert-base-uncased')
        self.vbert = tr.VisualBertModel.from_pretrained('uclanlp/visualbert-vcr-coco-pre')
        self.vbert.to(self.device)
        vis_emb_len = 0
        self.visual_embeds = torch.zeros((1,vis_emb_len,512), device=self.device)
        self.visual_attention_mask = torch.ones((1,vis_emb_len), device=self.device)

    def classify(self, example):
        svs = []
        for sent in example:
            inputs = self.tok(sent, return_tensors='pt')
            inputs.to(self.device)
            outputs = self.vbert(**inputs, visual_embeds=self.visual_embeds, visual_attention_mask=self.visual_attention_mask).last_hidden_state
            outputs = np.array(outputs.cpu().detach())
            outputs = np.sum(outputs, axis=1)
            svs.append(outputs)
        combinedv = np.append(svs[0], [svs[1]])
        combinedv = np.expand_dims(combinedv, axis=0)#combinedv.unsqueeze(0)
        outp = self.classifier.predict(combinedv)
        return outp
