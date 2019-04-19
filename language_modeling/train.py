import torch
import numpy as np
from text_utils import TextEncoder
from model_pytorch import *

class CustomLMModel(torch.nn.Module):
    """ Language model using transformer"""
    def __init__(self, cfg, vocab=40990, n_ctx=512, return_probs=True,
                 encoder_path='./model/encoder_bpe_40000.json', bpe_path='./model/vocab_40000.bpe'):
        super(CustomLMModel, self).__init__()
        # Transformer block
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        # Language modeling head to convert transformer output to word probabilities
        self.lm_head = LMHead(self.transformer, cfg, trunc_and_reshape=False)
        # Should the model return probabilities or without softmax
        self.return_probs = return_probs
        # Text encoder to convert word to index
        self.text_encoder = TextEncoder(encoder_path,bpe_path)
        
        if self.return_probs:
            pos_emb_mask = torch.zeros(1, 1, vocab)
            pos_emb_mask[:, :, -n_ctx:] = -1e12
            self.register_buffer('pos_emb_mask', pos_emb_mask)


    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        if self.return_probs:
            lm_logits = F.softmax(lm_logits + self.pos_emb_mask, dim=-1)
        return lm_logits

# Create model
lm_model = CustomLMModel(DEFAULT_CONFIG, return_probs=True)
# load pretrained weights
load_openai_pretrained_model(lm_model.transformer, n_ctx=512, n_special=0)
# save model for future use
torch.save(lm_model, './trained_lm_model')