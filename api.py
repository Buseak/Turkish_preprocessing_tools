from flask import Flask, json, g, request, jsonify, json
import tools.tokenizer as tokenizer, tools.sentence_splitter as sentence_splitter, tools.deasciifier as deasciifier, tools.pos_tagger as pos_tagger, tools.vowelizer as vowelizer, tools.spellcorrector as spellcorrector, tools.morphological as morphological
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

@app.route("/postagger", methods=["POST"])
def pos_tag():
    json_data = json.loads(request.data)
    postagger_instance = pos_tagger.PosTagger()
    response=postagger_instance.pos_tag(json_data['text'])

    result = {"Response": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/vowelizer", methods=["POST"])
def vowelize():
    json_data = json.loads(request.data)
    vowelizer_instance = vowelizer.Vowelizer()
    response=vowelizer_instance.vowelize(json_data['text'])

    result = {"Response": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/spellcorrector", methods=["POST"])
def correct_spelling():
    json_data = json.loads(request.data)
    spellcorrector_instance = spellcorrector.SpellCorrector()
    response=spellcorrector_instance.correct_spelling(json_data['text'])

    result = {"Response": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/morphological", methods=["POST"])
def morphologically_analyze():
    json_data = json.loads(request.data)
    morphological_instance = morphological.Morphological()
    response=morphological_instance.morphologic_analyze(json_data['text'])

    result = {"Response": response}
    response = app.response_class(
        response=json.dumps(result),
        status=200,
        mimetype='application/json'
    )
    return response



if __name__ == "__main__":
    app.run(host='0.0.0.0',threaded=False,)