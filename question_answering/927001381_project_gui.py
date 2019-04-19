#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tkinter

from tkinter import *
import numpy
import random
import torch
import os
import numpy as np
from modeling import BertForQuestionAnswering
from tokenization import BertTokenizer
from question_answering_model import question_answering
from download_utils import download_file_from_google_drive

def make_batch(X, n_vocab):
    X = np.array(X)
    assert X.ndim in [1, 2]
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    pos_enc = np.arange(n_vocab, n_vocab + X.shape[-1])
    pos_enc = np.expand_dims(pos_enc, axis=0)
    batch = np.stack([X, pos_enc], axis=-1)
    batch = torch.tensor(batch, dtype=torch.long).to(device)
    return batch

def append_batch(X, next_idx):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
    return torch.cat((X, next_x), 1)
    
def predict_next_word(text, gen_len=20, topk=10):
    generated_text = text
    n_vocab = len(lm_model.text_encoder.encoder)
    encoded_text = lm_model.text_encoder.encode([text,])
    encoded_text = make_batch(encoded_text, n_vocab)
    
    for _ in range(gen_len):
        lm_probs = lm_model(encoded_text)
        values, indices = lm_probs[:, -1, :].topk(topk)
        next_idx = indices.gather(-1, torch.multinomial(values, 1))
        next_token = lm_model.text_encoder.decoder[next_idx.item()].replace('</w>', '')
        generated_text += (' ' + next_token)
        encoded_text = append_batch(encoded_text, next_idx)
    return generated_text
        
def update_text(textbox, textvar):
    textbox.delete(0.0, tkinter.END)
    textbox.insert('insert', textvar+'\n')
    textbox.update()

# Function reads the text and question and answers
def answer():
    # Read the text from the entry widget
    doc_text = doc_var.get('1.0', 'end')
    question_text = question_var.get('1.0', 'end')
    # Generate single word
    textvar = question_answering(doc_text, question_text, model, tokenizer)
    # Update the entry
    update_text(answer1, textvar[0][0])
    update_text(answer2, textvar[1][0])
    update_text(answer3, textvar[2][0])

    update_text(confidence1, textvar[0][1])
    update_text(confidence2, textvar[1][1])
    update_text(confidence3, textvar[2][1])

CONFIG_FILE_ID = "1sfOG6bKLtLAhq5GoakkcbIfNIiCM0lZr"
MODEL_FILE_ID = "1ZjbUkhlpGOc1M4YssQ1YC29RaiAr31vM"
VOCAB_FILE_ID = "1bR7AnooSKRtrjL9GlqTJNiWyiZvQOaTD"

CONFIG_PATH = "./bert_squad/config.json"
MODEL_PATH = "./bert_squad/pytorch_model.bin"
VOCAB_PATH = "./bert_squad/vocab.txt"

print("Downloading model parameters...")
# Download model files
if not os.path.isfile(CONFIG_PATH): download_file_from_google_drive(CONFIG_FILE_ID, CONFIG_PATH)
if not os.path.isfile(MODEL_PATH):  download_file_from_google_drive(MODEL_FILE_ID, MODEL_PATH)
if not os.path.isfile(VOCAB_PATH):  download_file_from_google_drive(VOCAB_FILE_ID, VOCAB_PATH)

model = BertForQuestionAnswering.from_pretrained("./bert_squad/")
tokenizer = BertTokenizer.from_pretrained("./bert_squad/", do_lower_case=True)
device = "cpu"
model.to(device)

# Create TK instance
top = Tk()
top.title = 'Question Answering Model'
top.geometry('1400x700')

answer_dim = (35, 3)
confidence_dim = (35, 3)
text_dim = (120, 10)

# Text entry widget to enter text
doc_var=Text(top, width=text_dim[0], height=text_dim[1])
doc_var.tag_configure("center", justify='center')
doc_var.grid(row=0)
doc_var.insert('1.0', 'Enter passage here.')

question_var=Text(top, width=text_dim[0], height=text_dim[1])
question_var.tag_configure("center", justify='center')
question_var.grid(row=1)
question_var.insert('1.0', 'Enter question here.')

# "Answer" button: triggers the answer function
answer_button = Button(top, text ='Answer', command = answer)
answer_button.grid(row=2, column=0)


l1=Label(top,text='Please type the passage and question, then press Answer')
l1.grid(row=3)

answer1=Text(top, width=answer_dim[0], height=answer_dim[1])
answer1.grid(row=4, column=0)

answer2=Text(top, width=answer_dim[0], height=answer_dim[1])
answer2.grid(row=6, column=0)

answer3=Text(top, width=answer_dim[0], height=answer_dim[1])
answer3.grid(row=8, column=0)

confidence1=Text(top, width=confidence_dim[0], height=confidence_dim[1])
confidence1.grid(row=4, column=1)

confidence2=Text(top, width=confidence_dim[0], height=confidence_dim[1])
confidence2.grid(row=6, column=1)

confidence3=Text(top, width=confidence_dim[0], height=confidence_dim[1])
confidence3.grid(row=8, column=1)

top.mainloop()