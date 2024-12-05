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

# tokenizer_model = 'bpe'
wandb_api = "df1248450b282ba9bdaf39161311b2d5c72ccad0"
wandb.login(key=wandb_api)
wandb.init(
        name = f"finetune_mix_spa_nllb_def_v2",
        # name = "tiny",
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
# NEW_SPM_NAME = f"/home/aaditd/3_Capstone/Tokenizers/models/spm_nllb_mixtec_77k_{tokenizer_model}.model"

batch_size = 16
max_length = 128 # Mean token length was 65 something
# training_steps = 60000
num_epochs = 10
losses = []
MODEL_SAVE_PATH = f'/data/shire/data/aaditd/mixtec_models/mix_spa_nllb_def_v2'

print("*"*50)
print("Train_Batches: ", len(df_train)//batch_size) # 2053
print("Val_Batches: ", len(df_val)//batch_size) # 226
print("*"*50)

# exit(0)

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
tokenizer = NllbTokenizer.from_pretrained(model_name, cache_dir=cache_path)
# tokenizer = NllbTokenizer.from_pretrained(model_name, vocab_file=NEW_SPM_NAME, cache_dir=cache_path)



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
# stats.to_csv("statistics_new_tokenizer.csv")

# print(f"OLD TOKENIZER: {len(tokenizer_old)}")
print(f"DEFAULT TOKENIZER: {len(tokenizer)}")
print()

# (NLLB Tokenizer size is 256204)
# 931 added tokens for Unigram (New Tokenizer Size is 257135)
# 4678 added tokens for BPE (New Tokenizer Size is 260882)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_path)

print(type(model)) # <class 'transformers.models.m2m_100.modeling_m2m_100.M2M100ForConditionalGeneration'>

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

def get_sequential_batch_pairs(batch_size, data, index, randomize=True, direction="to"):
    
    # l1, l2 are the column names for the respective languages and code1, code2 are the language codes!
    if randomize:
        (l1, code1), (l2, code2) = random.sample(LANGS, 2) # Sample 2 items from LANGS
    
    else:
        if direction == "from": # From Spanish to Mixtec
            (l1, code1), (l2, code2) = ('translation', 'spa_Latn'), ('cleaned_transcription', 'mix_Latn')
        elif direction == "to": # Mixtec To Spanish
            (l1, code1), (l2, code2) = ('cleaned_transcription', 'mix_Latn'), ('translation', 'spa_Latn')
    
    xx, yy = [], []
    
    for i in range(index, index+batch_size):
        item = data.iloc[i]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    
    return xx, yy, code1, code2

def translation_loss_one_batch(model, tokenizer, source_texts, target_texts, source_lang_code, target_lang_code):
    
    tokenizer.src_lang = source_lang_code
    x = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
    tokenizer.src_lang = target_lang_code
    y = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)

    # -100 is ignored in the loss function! (don't want the model to learn padding tokens!)
    y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
                
    batch_loss = model(**x, labels=y.input_ids).loss
    
    return batch_loss

def train_loop(df_train, model, optimizer, scheduler, epoch_number):
    # Shuffle before the epoch!!
    df_train = df_train.sample(frac=1)
    
    model.train()
    x, y, loss = None, None, None

    train_loss = 0 # Epoch Train loss
    
    num_batches = len(df_train)//batch_size
    # print("Num Batches: ", num_batches)
    # print("DF_Train: ", len(df_train))
    tq = trange(num_batches)
    
    count_index = 0
    for i in tq:
        xx, yy, lang1, lang2 = get_sequential_batch_pairs(batch_size=batch_size, data=df_train, index=count_index)
        
        try:
            
            loss = translation_loss_one_batch(model, tokenizer, 
                                              source_texts=xx, 
                                              target_texts=yy,
                                              source_lang_code=lang1,
                                              target_lang_code=lang2)
            
            loss.backward()
            losses.append(loss.item())
            train_loss += loss.item()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            
        except RuntimeError as e: # OOM Error!
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            cleanup()
            print('error', max(len(s) for s in xx + yy), e)
            continue

        if i%1200 == 0:
            # print(i, np.mean(losses[-200:])) # Mean loss of the most recent 200 batches
            
            if i > 0:
                model.save_pretrained(MODEL_SAVE_PATH)
                tokenizer.save_pretrained(MODEL_SAVE_PATH)
                print(f"Saved Model and tokenizer after Batch {i}!!")
        
        count_index += batch_size
        
        wandb.log({"batch_train_loss": loss.item(), "num_train_examples": epoch_number*len(df_train) + (i+1) * batch_size})
    
    return train_loss/num_batches


def val_loop(df_val, model, epoch_number):
    # Shuffle the Val dataset!
    df_val = df_val.sample(frac=1)
    model.eval()
    
    num_batches = len(df_val)//batch_size
    
    tq = trange(num_batches)
    val_loss = 0 # Epoch Val loss
    count_index = 0
    
    with torch.no_grad():
        for i in tq:
            xx_to, yy_to, lang1_to, lang2_to = get_sequential_batch_pairs(batch_size=batch_size, data=df_val, index=count_index, randomize=False, direction="to")
            xx_from, yy_from, lang1_from, lang2_from = get_sequential_batch_pairs(batch_size=batch_size, data=df_val, index=count_index, randomize=False, direction="from")
            
            try:
                
                # Mixtec to Spanish Validation Loss
                to_loss = translation_loss_one_batch(model, tokenizer, 
                                                              source_texts=xx_to, 
                                                              target_texts=yy_to, 
                                                              source_lang_code=lang1_to, 
                                                              target_lang_code=lang2_to)
                
                # Spanish to Mixtec Validation Loss
                from_loss = translation_loss_one_batch(model, tokenizer, 
                                                              source_texts=xx_from, 
                                                              target_texts=yy_from, 
                                                              source_lang_code=lang1_from, 
                                                              target_lang_code=lang2_from)
                
                
                
                # Average them for epoch loss!
                val_loss += (to_loss.item() + from_loss.item())/2
                
                
            except RuntimeError as e: # OOM Error!
                x, y, loss = None, None, None
                cleanup()
                print('error', max(len(s) for s in xx_to + yy_to), e)
                continue
            
            count_index += batch_size
            
            wandb.log({"batch_val_loss_m2s": to_loss.item(), "batch_val_loss_s2m": from_loss.item(), "num_val_examples": epoch_number*len(df_val) + (i+1) * batch_size})

    return val_loss/num_batches

# df_small_train = df_train.head(144)
# df_small_val = df_val.head(64)

for epoch in range(num_epochs):
    # print(f"Epoch {epoch+1} --------------------------------\n")
    epoch_train_loss = train_loop(df_train, model, optimizer, scheduler, epoch)
    epoch_val_loss = val_loop(df_val, model, epoch)
    wandb.log({"epoch_train_loss":epoch_train_loss, "epoch":epoch})
    
    wandb.log({"epoch_val_loss":epoch_val_loss, "epoch":epoch})
    
    
wandb.finish()
        
print("STEP 5: TRAINING DONE!")