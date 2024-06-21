
from gensim.models import KeyedVectors
import json
from get_sentence_vector import get_sentence_vector

model_path = './custom_model.kv'


def _get_vectors_from_files(file_path):
    # Load the JSON array from a file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Loop through the strings in the JSON array and print them
    vectors = {}
    for string in data:
        key = '_'.join(string.lower().split())
        vectors[key] = get_sentence_vector(string)
    return vectors


def create_model(input_json_path):
    vectors = _get_vectors_from_files(input_json_path)
    vector_size = len(list(vectors.values())[0])
    model = KeyedVectors(vector_size)
    for key in vectors:
        model.add_vector(key, vectors[key])
    model.save(model_path)


def check_model():
    # Load the saved model
    model = KeyedVectors.load(model_path)

    # List all keys (vocabulary terms)
    keys = model.index_to_key

    # Access vectors for each key
    print("Vectors:")
    for key in keys:
        print(f"{key}: {model[key]}")



# create_model('./shop-item.json')

check_model()