from flask import Flask, json, g, request, jsonify, json
import tokenizer, sentence_splitter, deasciifier
app = Flask(__name__)

@app.route("/tokenizer", methods=["POST"])
def tokenize():
    json_data = json.loads(request.data)
    tokenizer_instance = tokenizer.Tokenizer()
    response=tokenizer_instance.tokenize(json_data['text'])

    result = {"tokens": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/sentencesplitter", methods=["POST"])
def split_sentences():
    json_data = json.loads(request.data)
    sentencesplitter_instance = sentence_splitter.SentenceSplitter()
    response=sentencesplitter_instance.split_sentences(json_data['text'])

    result = {"sentences": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/deasciifier", methods=["POST"])
def deasciify():
    json_data = json.loads(request.data)
    deasciifier_instance = deasciifier.Deasciifier()
    response=deasciifier_instance.deasciify(json_data['text'])

    result = {"Response": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=False,)