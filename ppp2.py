import numpy as np
import gensim
from gensim.models import KeyedVectors
import json
from openai import OpenAI


model_path = './custom_model.kv'

 
def get_sentence_vector_local(sentence):
    # Load the pre-trained Word2Vec model from the GoogleNews vectors file
    ref_model_path = './GoogleNews-vectors-negative300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(ref_model_path, binary=True)

    # Tokenize the sentence
    words = sentence.lower().split()

    # Vectorize the sentence
    return np.mean([model[word] for word in words if word in model], axis=0)


def get_sentence_vector_openai(sentence):
    client = OpenAI()
    response = client.embeddings.create(
        input=sentence,
        model="text-embedding-3-small"
    )
    embedded = response.data[0].embedding
    return np.array(embedded)


def get_sentence_vector(sentence):
    return get_sentence_vector_openai(sentence)

def get_vectors_from_files(file_path):
    # Load the JSON array from a file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Loop through the strings in the JSON array and print them
    vectors={}
    for string in data:
        key = '_'.join(string.lower().split())
        vectors[key] = get_sentence_vector(string)
    return vectors

def create_model(input_json_path):
    vectors = get_vectors_from_files(input_json_path)
    vector_size = len(list(vectors.values())[0])
    model = KeyedVectors(vector_size)
    for key in vectors:
        model.add_vector(key, vectors[key])
    model.save(model_path)


def find_similar(query):   
    model = KeyedVectors.load(model_path)
    query_vector = get_sentence_vector(query)
    # Find the top 5 most similar items to the query vector
    return model.similar_by_vector(query_vector, topn=3)


def check_model():
    # Load the saved model
    model = KeyedVectors.load(model_path)

    # List all keys (vocabulary terms)
    keys = model.index_to_key

    # Access vectors for each key
    print("Vectors:")
    for key in keys:
        print(f"{key}: {model[key]}")


def practical_example(input):
    items = find_similar(input);
    results = [item[0].replace("_", " ") for item in items]

    results_string = "\n".join(results)
     
    print("--------------------")
    print("suggestions: \n", results_string)
    print("--------------------\n\n")


    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are the shopkeeper and answer questions of customers best using  information below: \n ``` " + results_string + "```"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": input
            }
        ]
        }
    ]
    )

    return response.choices[0].message.content







input = "cold drink, with taste of fruits, and no sugar"

# uncomment to create the model
create_model("./shop-items.json")

# uncomment to check the model
#check_model()

# uncomment to find similar
#find_similar("I want to east something green")


# uncomment to run the practical example
print( practical_example(input))


