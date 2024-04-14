import pickle
import torch

class SpellCorrector:
  def __init__(self):
    self.model_first = pickle.load(open('tools/models/spellcorrector_model_1_pkl.pickle', 'rb'))
    self.model_first.eval()
    self.model_second = pickle.load(open('tools/models/spellcorrector_model_2_pkl.pickle', 'rb'))
    self.model_second.eval()
    self.tokenizer = pickle.load(open('tools/tokenizers/spellcorrector_tokenizer_pkl.pickle', 'rb'))
  
  def correct_spelling(self, sent):
    txt_with_dummy_char = self.add_dummy_char(sent)
    inputs = self.tokenizer(txt_with_dummy_char, add_special_tokens = True, return_tensors="pt")
    tokenized_inputs = self.tokenizer(txt_with_dummy_char, add_special_tokens = True)
    tokens = self.tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"])
    #inputs.to(device)
    with torch.no_grad():
        logits = self.model_first(**inputs).logits
        logits_2 = self.model_second(**inputs).logits

    new_logits = torch.cat((logits, logits_2), 2)
    predictions = torch.argmax(new_logits, dim=2)
    new_preds = predictions[0].to('cpu').numpy()
    for m in range(len(new_preds)):
        if new_preds[m] > 105:
            old_tensor = new_preds[m]
            new_idx = old_tensor -106
            new_preds[m] = new_idx
            #print(predictions[0][m])

    #predictions = torch.argmax(logits, dim=2)
    predicted_token_class = [self.model_first.config.id2label[t] for t in new_preds]
    tag_list = self.remove_special_tokens(predicted_token_class)
    tokens = self.remove_special_tokens(tokens)
    predicted_str = "".join(tag_list)
    cleared_sent = self.clear_sents(predicted_str)
    return cleared_sent

  def remove_special_tokens(self, tag_list):
    tag_list.pop(0)
    tag_list.pop(-1)
    return tag_list
  
  def add_dummy_char(self, sentence):
    char_list = list(sentence)
    return ("ß".join(char_list)+"ß")
  
  def clear_sents(self, sent):
    sent = sent.replace("ß","")
    return sent
