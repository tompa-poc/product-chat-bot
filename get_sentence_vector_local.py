import numpy as np
import gensim


def get_sentence_vector_local(sentence):
    # Load the pre-trained Word2Vec model from the GoogleNews vectors file
    ref_model_path = './GoogleNews-vectors-negative300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(ref_model_path, binary=True)

    # Tokenize the sentence
    words = sentence.lower().split()

    # Vectorize the sentence
    return np.mean([model[word] for word in words if word in model], axis=0)
