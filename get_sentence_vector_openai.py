from openai import OpenAI
import numpy as np
def get_sentence_vector_openai(sentence):
    client = OpenAI()
    response = client.embeddings.create(
        input=sentence,
        model="text-embedding-3-small"
    )
    embedded = response.data[0].embedding
    return np.array(embedded)
