from gensim.models import KeyedVectors
from openai import OpenAI
from get_sentence_vector import get_sentence_vector

model_path = './custom_model.kv'

def _find_similar(query):
    model = KeyedVectors.load(model_path)
    query_vector = get_sentence_vector(query)
    # Find the top 5 most similar items to the query vector
    return model.similar_by_vector(query_vector, topn=3)


def answer_question(input):
    items = _find_similar(input);
    results = [item[0].replace("_", " ") for item in items]

    results_string = "\n".join(results)

    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are the shopkeeper and answer questions of customers best you can based on these products prioritizing which makes most sense and in order : \n ``` " + results_string + "```"
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

    return {
        "response" : response.choices[0].message.content,
        "recommendations": results
    }
