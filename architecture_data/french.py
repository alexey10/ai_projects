import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

corpus = [
    "Ceci est une phrase.",
    "C'est un autre exemple de phrase.",
    "Voici une troisième phrase.",
    "Il fait beau aujourd'hui.",
    "J'aime beaucoup la cuisine française.",
    "Quel est ton plat préféré ?",
    "Je t'adore.",
    "Bon appétit !",
    "Je suis en train d'apprendre le français.",
    "Nous devons partir tôt demain matin.",
    "Je suis heureux.",
    "Le film était vraiment captivant !",
    "Je suis là.",
    "Je ne sais pas.",
    "Je suis fatigué après une longue journée de travail.",
    "Est-ce que tu as des projets pour le week-end ?",
    "Je vais chez le médecin cet après-midi.",
    "La musique adoucit les mœurs.",
    "Je dois acheter du pain et du lait.",
    "Il y a beaucoup de monde dans cette ville.",
    "Merci beaucoup !",
    "Au revoir !",
    "Je suis ravi de vous rencontrer enfin !",
    "Les vacances sont toujours trop courtes.",
    "Je suis en retard.",
    "Félicitations pour ton nouveau travail !",
    "Je suis désolé, je ne peux pas venir à la réunion.",
    "À quelle heure est le prochain train ?",
    "Bonjour !",
    "C'est génial !"
]

# Tokenizer
tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# Build vocabulary from corpus
def yield_tokens(data_iter):
    for sentence in data_iter:
        yield tokenizer(sentence)

vocab = build_vocab_from_iterator(yield_tokens(corpus), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Collate function using tokenizer and vocab
def collate_fn_fr(batch):
    tensor_batch = []
    for sentence in batch:
        tokens = tokenizer(sentence)
        indices = [vocab[token] for token in tokens]
        tensor_batch.append(torch.tensor(indices, dtype=torch.long))
    padded_batch = pad_sequence(tensor_batch, batch_first=True, padding_value=vocab["<unk>"])
    return padded_batch

# Sort corpus by length
sorted_data = sorted(corpus, key=lambda x: len(tokenizer(x)))

# Create DataLoader
dataloader = DataLoader(sorted_data, batch_size=4, shuffle=False, collate_fn=collate_fn_fr)

# Print padded batches
for batch in dataloader:
    print(batch)

