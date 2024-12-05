from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import random
import re
from tqdm.auto import tqdm, trange

df_train = pd.read_json("../Data/train.json", orient="records", lines=True)
df_test = pd.read_json("../Data/test.json", orient="records", lines=True)
df_val = pd.read_json("../Data/validation.json", orient="records", lines=True)
print(df_train.shape) # (32849, 6)
print(df_test.shape) # (2005, 6)
print(df_val.shape) # (3630, 6)
print(df_test.columns) # ['doc', 'sid', 'speaker', 'transcription', 'cleaned_transcription', 'translation']


model_name = "facebook/nllb-200-distilled-600M"
cache_path = "/data/shire/data/aaditd/trial/"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_path)

tokenizer.src_lang = "spa_Latn"
text="escucha las palabras de las brujas, los secretos escondidos en la noche, los antiguos dioses invocamos ahora la obra de la magia oculta"
inputs = tokenizer(text="escucha las palabras de las brujas, los secretos escondidos en la noche, los antiguos dioses invocamos ahora la obra de la magia oculta", return_tensors="pt")


def word_tokenize_m(text):
    """
    Split a text into words, numbers, and punctuation marks,
    splitting only on whitespaces, '=' and '-'.
    """
    # Split on spaces, '=' or '-' and capture words, numbers, and punctuation
    return re.findall(r'[\w\(\)\.,!?]+', text.replace('=', ' ').replace('-', ' '))

def word_tokenize_s(text):
    """
    Split a text into words, numbers, and punctuation marks
    (for languages where words are separated by spaces)
    """
    return re.findall('(\w+|[^\w\s])', text)

sample = df_train.sample(20000, random_state=1)
sample["Mixtec_Tokens"] = sample.cleaned_transcription.apply(tokenizer.tokenize)
sample["Spanish_Tokens"] = sample.translation.apply(tokenizer.tokenize)
sample["Mixtec_Human_Words"] = sample.cleaned_transcription.apply(word_tokenize_m)
sample["Spanish_Human_Words"] = sample.translation.apply(word_tokenize_s)

# print(sample.head(10))

stats = sample[['Mixtec_Tokens', 'Spanish_Tokens', 'Mixtec_Human_Words', 'Spanish_Human_Words']].applymap(len).describe()

print(stats.Spanish_Tokens['mean'] / stats.Spanish_Human_Words['mean']) # 21.6 / 17.3 = 1.25
print(stats.Mixtec_Tokens['mean']/stats.Mixtec_Human_Words['mean']) # 65.6 / 22.0 = 2.97

# 3 tokens per Mixtec word -> Translation quality might be fine without even extending the vocab!
# stats.to_csv("statistics.csv")
# print(stats)

texts_with_unk = [text for text in tqdm(df_train.cleaned_transcription) if tokenizer.unk_token_id in tokenizer(text).input_ids]
print(texts_with_unk) # ZERO!!
