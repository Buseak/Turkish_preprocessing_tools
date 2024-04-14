import pickle
import torch

class Tokenizer:
  def __init__(self):
    self.model = pickle.load(open('tools/models/tokenizer_model_pkl.pickle', 'rb'))
    self.tokenizer = pickle.load(open('tools/tokenizers/tokenizer_canine.pickle', 'rb'))

  def tokenize(self, sent):
    predicted_tags = self.predict_tags(sent)
    tokens = self.get_tokens(predicted_tags, sent)
    return tokens
  
  def predict_tags(self, sent):
    inputs = self.tokenizer(sent, add_special_tokens = True, return_tensors="pt")
    with torch.no_grad():
        logits = self.model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0]]
    tag_list = self.remove_special_tokens(predicted_token_class)

    return tag_list
  
  def get_tokens(self, predicted_tags, input_sent):
    sentence = input_sent
    pred_tags = predicted_tags

    tokens_list = []
    token = ""
    for j in range(len(pred_tags)):
      label = pred_tags[j]
      if j == (len(pred_tags) - 1):
        if label == "B":
          if token != "":
            tokens_list.append(token)
          token = sentence[j]
          tokens_list.append(token)
        else:
          token += sentence[j]
          tokens_list.append(token)

      else:

        if j == 0:
          if label == "B":
            token += sentence[j]
        else:
          if label == "B":
            if token != "":
              tokens_list.append(token)
            token = sentence[j]
          if label == "I":
            token += sentence[j]

          if label == "N":
            tokens_list.append(token)
            token = ""

    return tokens_list

  def remove_special_tokens(self, tag_list):
    tag_list.pop(0)
    tag_list.pop(-1)
    return tag_list

