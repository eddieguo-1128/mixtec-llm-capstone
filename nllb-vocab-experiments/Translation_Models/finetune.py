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
import wandb

# WANDB LOGIN
# _________________________________________________________________________________________________________

tokenizer_model = 'unigram'
wandb_api = "df1248450b282ba9bdaf39161311b2d5c72ccad0"
wandb.login(key=wandb_api)
wandb.init(
        name = f"finetune_mix_spa_nllb_{tokenizer_model}",
        reinit = True,
        project = "Mixtec_LLM",
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# DATA and HYPERPARAMETERS
# _________________________________________________________________________________________________________

df_train = pd.read_json("../Data/train.json", orient="records", lines=True)
df_test = pd.read_json("../Data/test.json", orient="records", lines=True)
df_val = pd.read_json("../Data/validation.json", orient="records", lines=True)

model_name = "facebook/nllb-200-distilled-600M"
cache_path = "/data/shire/data/aaditd/trial/"
NEW_SPM_NAME = f"/home/aaditd/3_Capstone/Tokenizers/models/spm_nllb_mixtec_77k_{tokenizer_model}.model"

batch_size = 16
max_length = 128 # Mean token length was 65 something
training_steps = 60000
losses = []
MODEL_SAVE_PATH = f'/data/shire/data/aaditd/mixtec_models/mix_spa_nllb_{tokenizer_model}_v1'


# HELPER FUNCTIONS
# _________________________________________________________________________________________________________
def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def get_non_printing_char_replacer(replace_by: str = " "):
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")

def preproc(text):
    clean = replace_nonprint(text)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean

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

# STEP 3: EXPANDING THE VOCABULARY
# _________________________________________________________________________________________________________
tokenizer_old = NllbTokenizer.from_pretrained(model_name, cache_dir=cache_path)
tokenizer = NllbTokenizer.from_pretrained(model_name, vocab_file=NEW_SPM_NAME, cache_dir=cache_path)

fix_tokenizer(tokenizer)

# STEP 3.5 STATISTICS
# _________________________________________________________________________________________________________
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

print(stats.Spanish_Tokens['mean'] / stats.Spanish_Human_Words['mean']) # 20.89 / 17.3 = 1.21
print(stats.Mixtec_Tokens['mean']/stats.Mixtec_Human_Words['mean']) # 60.5 / 22.0 = 2.75

# 2.75 tokens per Mixtec word -> Translation quality might be fine without even extending the vocab!
stats.to_csv("statistics_new_tokenizer.csv")

print(f"OLD TOKENIZER: {len(tokenizer_old)}")
print(f"NEW TOKENIZER: {len(tokenizer)}")
print()

added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
print(f"ADDED VOCAB: {len(added_vocab)}")

# (NLLB Tokenizer size is 256204)
# 931 added tokens for Unigram (New Tokenizer Size is 257135)
# 4678 added tokens for BPE (New Tokenizer Size is 260882)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_path)

print(type(model)) # <class 'transformers.models.m2m_100.modeling_m2m_100.M2M100ForConditionalGeneration'>
model.resize_token_embeddings(len(tokenizer))

# Initialize embeddings of new tokens as the average of the old tokens that corresponded to it
for t in tqdm(added_vocab):
    
    # t is a string, tt is a token_id!! (Recall Incantations)
    
    # Old tokens that corresponded to the tokens we're adding
    tt = tokenizer_old(t, add_special_tokens=False).input_ids
    
    # if there are none, assign it the UNK token from the old tokenizer!
    if len(tt) == 0:
        tt = [tokenizer_old.unk_token_id]
    
    idx = tokenizer.convert_tokens_to_ids(t)  # t is a string, tt is a token_id!! (Recall Incantations)
    
    # model.shared contains the embeddings for the tokens!
    # We want to initialize the embeddings of the new tokens as the mean of the old tokens (that would've corresponded to them!)
    model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)

print(f"STEP 3: {tokenizer_model} EMBEDDINGS RESIZING AND REINITIALIZATION DONE!")

# STEP 5: TRAINING THE MODEL
# _________________________________________________________________________________________________________

model.cuda()
print(f"MODEL IS USING DEVICE: {model.device}")
optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3
)

scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000)

LANGS = [('translation', 'spa_Latn'), ('cleaned_transcription', 'mix_Latn')]

# For each training batch, we randomly choose the translation direction and sample sentence pairs

def get_batch_pairs(batch_size, data=df_train):
    
    # l1, l2 are the column names for the respective languages and code1, code2 are the language codes!
    (l1, code1), (l2, code2) = random.sample(LANGS, 2) # Sample 2 items from LANGS
    
    xx, yy = [], []
    
    for _ in range(batch_size):
        rng = random.randint(0, len(data)-1)
        item = data.iloc[rng]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    
    return xx, yy, code1, code2

# print(get_batch_pairs(5))

model.train()
x, y, loss = None, None, None

tq = trange(len(losses), training_steps)
for i in tq:
    xx, yy, lang1, lang2 = get_batch_pairs(batch_size=batch_size)
    
    try:
        tokenizer.src_lang = lang1
        x = tokenizer(xx, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
        tokenizer.src_lang = lang2
        y = tokenizer(yy, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)

        # -100 is ignored in the loss function! (don't want the model to learn padding tokens!)
        y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
        
        loss = model(**x, labels=y.input_ids).loss
        
        loss.backward()
        losses.append(loss.item())
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        
        
    except RuntimeError as e: # OOM Error!
        optimizer.zero_grad(set_to_none=True)
        x, y, loss = None, None, None
        cleanup()
        print('error', max(len(s) for s in xx + yy), e)
        continue

    if i%1000 == 0:
        print(i, np.mean(losses[-1000:])) # Mean loss of the most recent 1000 iterations
        
        if i > 0:
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            print(f"Saved Model and tokenizer at iteration {i}!!")
    
    
    wandb.log({"train_loss": loss, "iteration": i})
    
wandb.finish()
        
print("STEP 5: TRAINING DONE!")