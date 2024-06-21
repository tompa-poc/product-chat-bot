from flask import Flask, request, jsonify

from answer_question import answer_question

app = Flask(__name__)

@app.route('/', methods=['POST'])
def greet():
    try:
            data = request.get_json()
            query = data.get('query')
            answer = answer_question(query)
            return jsonify(answer)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

# example curl request
# curl -X POST -H "Content-Type: application/json" -d '{"query": "something expensive to impress my girlfriend, she likes sweet alcoholic drinks"}' http://127.0.0.1:5000/
