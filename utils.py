import nltk
import numpy as np      
from nltk.stem.porter import PorterStemmer

def tokenize(sentance):
 
    return nltk.word_tokenize(sentance)

stemmer = PorterStemmer()
def stem(word):
  
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sents, all_words):
   
    sentence_words = [stem(word) for word in tokenized_sents]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in sentence_words: 
            bag[idx] = 1
    
    return bag

