#Tokenization Implementation

import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
import spacy
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from transformers import BertTokenizer
from transformers import XLNetTokenizer

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

text = "This is a sample sentence for word tokenization."
tokens = word_tokenize(text)
print(tokens)

# This showcases the use of the 'spaCy' tokenizer with torchtext's get_tokenizer function

text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# Making a list of the tokens and priting the list
token_list = [token.text for token in doc]
print("Tokens:", token_list)

# Showing token details
for token in doc:
    print(token.text, token.pos_, token.dep_)

text = "Unicorns are real. I saw a unicorn yesterday."
token = word_tokenize(text)
print(token)

