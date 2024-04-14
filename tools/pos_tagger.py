import pickle
import torch

class PosTagger:
  def __init__(self):
    self.model = pickle.load(open('tools/models/postagger_model_pkl.pickle', 'rb'))
    self.model.eval()
    self.tokenizer = pickle.load(open('tools/tokenizers/postagger_tokenizer_pkl.pickle', 'rb'))
    self.unique_pos_tags = ['INTJ',
                            'PART',
                            'VERB',
                            'ADP',
                            'ADJ',
                            'X',
                            'PUNCT',
                            'NOUN',
                            'NUM',
                            'ADV',
                            'CCONJ',
                            'SCONJ',
                            'DET',
                            'AUX',
                            'PRON',
                            'PROPN']
    self.id2label = self.create_id2label()

  def pos_tag(self, sent):
    sentence = sent.split()
    tag_list,tokens = self.predict_tags(sentence)
    return {"tokens":tokens,
            "tag_list":tag_list}
  
  def predict_tags(self, sent):
    inputs = self.tokenizer(sent, return_tensors="pt", is_split_into_words=True)
    tokenized_inputs = self.tokenizer(sent, is_split_into_words=True)
    tokens = self.tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
    with torch.no_grad():
        logits = self.model(**inputs).logits
    predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [t.item() for t in predictions[0]]
    subword_indexes = self.find_subword_indexes(tokens)
    tag_list = self.remove_by_indices(predicted_token_class, subword_indexes)
    tag_list = self.remove_special_tokens(tag_list)
    tokens = self.remove_by_indices(tokens, subword_indexes)
    tokens = self.remove_special_tokens(tokens)

    for i in range(len(tag_list)):
        tag_list[i] = self.id2label[tag_list[i]]

    return tag_list,tokens

  def remove_special_tokens(self, tag_list):
    tag_list.pop(0)
    tag_list.pop(-1)
    return tag_list
  
  def find_subword_indexes(self, token_list):
    subword_indexes = []
    for i in range(len(token_list)):
        if token_list[i][0] == '#':
            subword_indexes.append(i)
    return subword_indexes
  
  def create_id2label(self):
    id2label = {}
    for i in range(len(self.unique_pos_tags)):
        id2label[i]=self.unique_pos_tags[i]
    return id2label
  
  def remove_by_indices(self, iter, idxs):
    return [e for i, e in enumerate(iter) if i not in idxs]