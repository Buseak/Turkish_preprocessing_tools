import pickle
import torch

class Vowelizer:
  def __init__(self):
    self.model = pickle.load(open('tools/models/vowelizer_model_pkl.pickle', 'rb'))
    self.model.eval()
    self.tokenizer = pickle.load(open('tools/tokenizers/vowelizer_tokenizer_pkl.pickle', 'rb'))


  def vowelize(self, sent):
    input_sent_with_dummy_chars = self.add_dummy_char(sent)
    return self.get_sent(self.predict_tags(input_sent_with_dummy_chars), input_sent_with_dummy_chars)
  
  def predict_tags(self, sent):
    inputs = self.tokenizer(sent, add_special_tokens = True, return_tensors="pt")
    with torch.no_grad():
        logits = self.model(**inputs).logits

    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0]]
    tag_list = self.remove_special_tokens(predicted_token_class)

    return tag_list
  
  def get_sent(self, predicted_sent, input_sent):
    pred_tags = predicted_sent
    errored_sent = input_sent
    cleared_sent = []
    for ch in range(len(pred_tags)):
        if pred_tags[ch] == "NaN":
            cleared_sent.append(errored_sent[ch])
        else:
            cleared_sent.append(pred_tags[ch])
    new_sent = "".join(cleared_sent)
    sent_without_dummy = new_sent.replace("ß", "")
    return sent_without_dummy

  def remove_special_tokens(self, tag_list):
    tag_list.pop(0)
    tag_list.pop(-1)
    return tag_list

  def add_dummy_char(self, sentence):
    char_list = list(sentence)
    return ("ß".join(char_list)+"ß")