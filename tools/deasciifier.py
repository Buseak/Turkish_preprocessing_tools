import pickle
import torch

class Deasciifier:
  def __init__(self):
    self.model = pickle.load(open('tools/models/deasciifier_model_pkl.pickle', 'rb'))
    self.model.eval()
    self.tokenizer = pickle.load(open('tools/tokenizers/deasciifier_tokenizer_pkl.pickle', 'rb'))

  def deasciify(self, sent):
    predicted_tags = self.predict_tags(sent)
    result = self.get_sent(predicted_tags, sent)
    return result
  
  def predict_tags(self, sent):
    inputs = self.tokenizer(sent, add_special_tokens = True, return_tensors="pt")
    with torch.no_grad():
        logits = self.model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0]]
    tag_list = self.remove_special_tokens(predicted_token_class)

    return tag_list
  
  def get_sent(self, predicted_tags, input_sent):
    sentence_text = input_sent
    preds = predicted_tags
    converted_sent = []
    for j in range(len(preds)):
        if preds[j] == "NaN":
            converted_sent.append(sentence_text[j])
        else:
            converted_sent.append(preds[j])
    result = "".join(converted_sent)
    return result

  def remove_special_tokens(self, tag_list):
    tag_list.pop(0)
    tag_list.pop(-1)
    return tag_list