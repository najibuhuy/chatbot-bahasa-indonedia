import numpy as np
import nltk
# nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


factory = StemmerFactory()
stemmer = factory.create_stemmer()

#memisahkan setiap kata pada sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#mengambil kata dasar/awal dari array kata
def stem(word):
    return stemmer.stem(word.lower())

#membuat pola array binary dari tokenized_sentence
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag