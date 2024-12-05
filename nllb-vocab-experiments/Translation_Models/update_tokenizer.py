import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import NllbTokenizer
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


model_name = "facebook/nllb-200-distilled-600M"
tokenizer_model = 'unigram'
tokenizer_model_name = f"/home/aaditd/3_Capstone/Tokenizers/models/{tokenizer_model}.model"
tokenizer_vocab_name = f"/home/aaditd/3_Capstone/Tokenizers/models/{tokenizer_model}.vocab"
cache_path = "/data/shire/data/aaditd/trial/"

tokenizer = NllbTokenizer.from_pretrained(model_name, cache_dir=cache_path)

sp_trained = spm.SentencePieceProcessor(model_file=tokenizer_model_name)

added_spm = sp_pb2_model.ModelProto()
added_spm.ParseFromString(sp_trained.serialized_model_proto())

old_spm = sp_pb2_model.ModelProto()
old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

nllb_tokens_set = {p.piece for p in old_spm.pieces}
added_tokens_set = {p.piece for p in added_spm.pieces}

def write_list_to_file(list_to_write, filename):
    """Writes a list to a text file, one item per line.

    Args:
        list_to_write (list): The list to be written to the file.
        filename (str): The name of the file to write to.
    """

    with open(filename, 'w') as f:
        for item in list_to_write:
            f.write(str(item) + '\n')
            
write_list_to_file(list(nllb_tokens_set), '/home/aaditd/3_Capstone/Tokenizers/models/nllb.vocab')
write_list_to_file(list(added_tokens_set), f'/home/aaditd/3_Capstone/Tokenizers/models/added_{tokenizer_model}.vocab')
prev_min_score = old_spm.pieces[-1].score # Arranged in descending order

# Enrich the NLLB Tokenizer with all the new tokens from our Custom Mixtec SentencePiece model!
\
for p in added_spm.pieces:
    piece = p.piece
    
    if p.type != 1: # Type 2 is for <unk>, and Type 3 is for <s> and </s>
        continue

        
    
    if piece not in nllb_tokens_set:
        
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = p.score + prev_min_score # Set lower score for new tokens!
        old_spm.pieces.append(new_p)
    
NEW_SPM_NAME = f"/home/aaditd/3_Capstone/Tokenizers/models/spm_nllb_mixtec_77k_{tokenizer_model}.model"

with open(NEW_SPM_NAME, 'wb') as f:
    f.write(old_spm.SerializeToString())

print(f"{tokenizer_model} VOCAB UPDATE DONE!")