import pickle

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_spell = AutoModelForSeq2SeqLM.from_pretrained("Buseak/md_mt5_0109_v8")

tokenizer = AutoTokenizer.from_pretrained("Buseak/md_mt5_0109_v8")

import pickle
# create an iterator object with write permission - model.pkl
with open('morphological_analyzer_disambiguator_model_pkl.pickle', 'wb') as files:
    pickle.dump(model_spell, files)



with open('morphological_tokenizer_pkl.pickle', 'wb') as files:
    pickle.dump(tokenizer, files)

