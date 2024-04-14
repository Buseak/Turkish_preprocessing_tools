import pickle

from transformers import AutoTokenizer

tokenizer_canine = AutoTokenizer.from_pretrained("Buseak/canine_deasciifier_0305")

from transformers import AutoModelForTokenClassification

model_deasciifier = AutoModelForTokenClassification.from_pretrained("Buseak/canine_deasciifier_0305")

import pickle
# create an iterator object with write permission - model.pkl
with open('deasciifier_model_pkl.pickle', 'wb') as files:
    pickle.dump(model_deasciifier, files)

with open('deasciifier_tokenizer_pkl.pickle', 'wb') as files:
    pickle.dump(tokenizer_canine, files)

