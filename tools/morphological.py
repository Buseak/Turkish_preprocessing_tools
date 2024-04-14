import pickle
import torch
import re

class Morphological:
  def __init__(self):
    self.model = pickle.load(open('tools/models/morphological_analyzer_disambiguator_model_pkl.pickle', 'rb'))
    self.model.eval()
    self.tokenizer = pickle.load(open('tools/tokenizers/morphological_tokenizer_pkl.pickle', 'rb'))
    self.tag_list = ['z','YAmA','lAn','t','ByDoingSo',"\'",'Pers','DAn','m','lAş','NoHats','CH','mAktA','Ar','Hn','Num',
                    'Noun',
                    'A3sg',
                    'SH',
                    'P3sg',
                    'NDA',
                    'Loc',
                    'YHz',
                    'Verb',
                    'Pres',
                    'A1pl',
                    'Punc',
                    'Adv',
                    'Pron',
                    'Reflex',
                    'A1sg',
                    'Hm',
                    'P1sg',
                    'NDAn',
                    'Abl',
                    'Pnon',
                    'Nom',
                    'Det',
                    'DA',
                    'YHm',
                    'Pos',
                    'Hr',
                    'Aor',
                    'YsA',
                    'Cond',
                    'Imp',
                    'sHn',
                    'DHk',
                    'Adj',
                    'PastPart',
                    'NHn',
                    'Gen',
                    'NA',
                    'Dat',
                    'NH',
                    'Acc',
                    'YAn',
                    'PresPart',
                    'Prop',
                    'A3pl',
                    'YH',
                    'YDH',
                    'Past',
                    'Unknown',
                    'Apos',
                    'YA',
                    'YHp',
                    'mA',
                    'Inf2',
                    'mHş',
                    'Narr',
                    'Rüzgar',
                    'With',
                    'Hyor',
                    'Prog1',
                    'Interj',
                    'YlA',
                    'Ins',
                    'YAcAk',
                    'FutPart',
                    'Ques',
                    'Conj',
                    'Become',
                    'DHr',
                    'Caus',
                    'YArAk',
                    'HmHz',
                    'P1pl',
                    'Rel',
                    'Hl',
                    'Pass',
                    'DH',
                    'mAk',
                    'Inf1',
                    'Acro',
                    'lHk',
                    'FitFor',
                    'Fut',
                    'Neg',
                    'Postp',
                    'PCNom',
                    'Demons',
                    'HnHz',
                    'P2pl',
                    'Ness',
                    'PCAbl',
                    'lArH',
                    'P3pl',
                    'PCDat',
                    'Able',
                    'Agt',
                    'Quant',
                    'nHz',
                    'A2pl',
                    'YHş',
                    'Inf3',
                    'P2sg',
                    'şikayet',
                    'YHcH',
                    'Equ',
                    'Cop',
                    'sHz',
                    'Without',
                    'YmHş',
                    'PCIns',
                    'While',
                    'YHn',
                    'mAlH',
                    'Neces',
                    'Opt',
                    'sHnHz',
                    'Abbr',
                    'Desr',
                    'Acquire',
                    'CAsHnA',
                    'A2sg',
                    'ya',
                    'YHncA',
                    'When',
                    'lHm',
                    'Dup',
                    'Prog2',
                    'mAzlHk',
                    'NotState',
                    'YHver',
                    'Hastily',
                    'DHkçA',
                    'AsLongAs',
                    'gen',
                    'mHşlHk',
                    'NarrNess',
                    'AfterDoingSo',
                    'Yken',
                    'sA',
                    'lH',
                    'YAbil', 'lAr']


  def morphologic_analyze(self, text):
    preprocessed_text = self.preprocess(text)
    inputs = self.tokenizer(preprocessed_text, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = self.model.generate(inputs, max_new_tokens=1024, do_sample=True, top_k=40, top_p=0.95)
        decoded_outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    postprocessed_output = self.postprocess(decoded_outputs)
    ind_list = self.create_index_list(postprocessed_output)
    sent_dict_preds = self.convert_to_dict(ind_list, postprocessed_output)
    return sent_dict_preds

  def remove_special_tokens(self, tag_list):
    tag_list.pop(0)
    tag_list.pop(-1)
    return tag_list
  
  def preprocess(self, sentence):
    sentence = re.sub("([.:!?\'\-()])", r' \1 ', sentence)
    sentence = re.sub('(\")', r' \1 ', sentence)
    sentence = re.sub('\s{2,}', ' ', sentence)
    if sentence[0] == " ":
       sentence = sentence [1:]
    return sentence
  
  def postprocess(self, model_output):
    pattern = '('+'|'.join(self.tag_list) + ')'
    checks = ["z", "m", "t", "ya"]
    preds = model_output.split()
    for m in range(len(preds)):
        if preds[m] not in self.tag_list:
            result = re.match(r""+pattern+"", preds[m])
        if result != None:
            groups = result.groups()
            if groups[0] not in checks:
                group = groups[0]
                new_str = preds[m].replace(group, group+" ")
                preds[m] = new_str
        if m == len(preds)-1 and preds[m] == 'Punc' and len(preds[m-1])>1:
                if "." == preds[m-1][-1] or "?" == preds[m-1][-1] or "!" == preds[m-1][-1]:
                    my_punct = preds[m-1][-1]
                    preds[m-1] = preds[m-1][0:-1]
                    preds[m] = my_punct
                    preds.append("Punc")
    return " ".join(preds)
  
  def create_index_list(self, postprocessed_output):
    pred_list = []
    feats = postprocessed_output.split()
    for j in range(len(feats)):
        if feats[j] not in self.tag_list:
            pred_list.append(j)
    return pred_list
  
  def convert_to_dict(self, pred_index_list, postprocessed_output):
    sent_dict_preds = {}
    for m in range(len(pred_index_list)):

        ind = pred_index_list[m]
        pred_words = postprocessed_output.split()
        word = pred_words[ind]
        word = word + "_" + str(m)

        if ind < len(pred_words)-1 and (m+1<= len(pred_index_list)-1):
            next_ind = pred_index_list[m+1]
            sent_dict_preds[word] = pred_words[ind+1:next_ind]
        else:
            sent_dict_preds[word] = pred_words[ind+1:]
    return sent_dict_preds
  
  

 