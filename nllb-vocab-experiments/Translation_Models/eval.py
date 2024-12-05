import re
import sys
import gc
import random
import numpy as np
import unicodedata
from tqdm.auto import tqdm, trange
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from transformers import NllbTokenizer
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
import pandas as pd
from comet import download_model, load_from_checkpoint

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

def fix_tokenizer(tokenizer, new_lang='mix_Latn'):
    """
    Add a new language token to the tokenizer vocabulary 
    (this should be done each time after its initialization)
    """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {} 


# DATA and HYPERPARAMETERS
# _________________________________________________________________________________________________________

df_train = pd.read_json("../Data/train.json", orient="records", lines=True)
df_test = pd.read_json("../Data/test.json", orient="records", lines=True)
df_val = pd.read_json("../Data/validation.json", orient="records", lines=True)

# tokenizer_model = 'bpe'
tokenizer_model = 'def'
model_name = "facebook/nllb-200-distilled-600M"
cache_path = "/data/shire/data/aaditd/trial/"

model_load_name = f'/data/shire/data/aaditd/mixtec_models/mix_spa_nllb_{tokenizer_model}_v2'


model_oob = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_path).cuda()
tokenizer_oob = NllbTokenizer.from_pretrained(model_name, cache_dir=cache_path)

model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).cuda()
tokenizer = NllbTokenizer.from_pretrained(model_load_name)


# model = model_oob
# tokenizer = tokenizer_oob

fix_tokenizer(tokenizer)


def translate(
    text,
    src_lang='mix_Latn',
    tgt_lang='spa_Latn',
    a=32,
    b=3,
    max_input_length=1024,
    num_beams=4,
    **kwargs
):
    """ Turn a list of texts into a list of translations"""
    
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
    
    model.eval()
    
    result = model.generate(**inputs.to(model.device),
                            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                            max_new_tokens=int(a+b*inputs.input_ids.shape[1]),
                            num_beams=num_beams, **kwargs
                            
                            )
    
    return tokenizer.batch_decode(result, skip_special_tokens=True)

def batched_translate(texts, batch_size=16, **kwargs):
    """Translate texts in batches of similar length"""
    
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    
    for i in trange(0, len(texts2), batch_size):
        results.extend(translate(texts2[i:i+batch_size], **kwargs))
    
    return results

# TRANSLATION

df_test = df_test
mixtec_sentences = df_test["cleaned_transcription"]
spanish_sentences = df_test["translation"]

translated_spanish_sentences = batched_translate(texts=mixtec_sentences, src_lang='mix_Latn', tgt_lang='spa_Latn')
df_test["mix_to_spa"] = translated_spanish_sentences
translated_mixtec_sentences = batched_translate(texts=spanish_sentences, src_lang='spa_Latn', tgt_lang='mix_Latn')
df_test["spa_to_mix"] = translated_mixtec_sentences

df_test.to_csv(f"outputs/translations_unigram_oob.csv")

# Example
# mixtec_sentence = "tan3 ti1xin3=a2 chi1tun3 kan4 i13xa3=ra2 ji'4in4 kum3pa4ri2=ra1 tan3, tan3 ta1 nda'4bi2 inocente kan4 ndi4"
# reference_translation = "le hizo una maldad a su pobre compadre, y ese pobre inocente"
# print(f"REFERENCE TRANSLATION: {reference_translation}")
# print(f"MODEL TRANSLATION: {translate(text=mixtec_sentence)}")

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
import pandas as pd


def clean_result(df):
    return df.dropna()

# METRICS COMPUTATION

def corpus_performance(sys, ref, comet=False, source=None, comet_model=None):
    bleu = BLEU()
    chrf = CHRF()

    result_bleu = bleu.corpus_score(sys, [ref])
    result_chrf = chrf.corpus_score(sys, [ref])

    if comet:
      data = [{"src": source_sen, "mt":sys_sen, "ref":ref_sen} for source_sen, sys_sen, ref_sen in zip(source, sys, ref)]
      model_output = comet_model.predict(data, batch_size=8, gpus=1)
      print(f'COMET score: {round(model_output.system_score, 2)}')

    print(f'BLEU score: {round(result_bleu.score, 2)}')
    print(f'CHRF score: {round(result_chrf.score, 2)}')

def bidirectional_performance(df, comet_model):
  
  df_test = df
  df_test = clean_result(df_test)

  # METRICS COMPUTATION
  mix_truth = df_test["cleaned_transcription"].to_list()
  spa_truth = df_test["translation"].to_list()
  spa_model = df_test["mix_to_spa"].to_list()
  mix_model = df_test["spa_to_mix"].to_list()
  print(len(mix_truth))


  print('=' * 50)
  # Direction 1: Mixtec to Spanish
  print("MIXTEC TO SPANISH PERFORMANCE: ")
  # print(type([[s] for s in spa_truth]))
  corpus_performance(spa_model, spa_truth, comet=True, source=mix_truth, comet_model=comet_model)
  # print()
  # corpus_performance(spa_model, spa_truth, use_old)
  print('-' * 30)
  print("SPANISH TO MIXTEC PERFORMANCE: ")
  corpus_performance(mix_model, mix_truth, comet=True, source=spa_truth, comet_model=comet_model)
  print()
  print()
  

bidirectional_performance(df_test, comet_model)