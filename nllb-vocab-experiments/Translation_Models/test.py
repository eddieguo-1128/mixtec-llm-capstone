from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/nllb-200-distilled-600M"
cache_path = "/data/shire/data/aaditd/trial/"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_path)

tokenizer.src_lang = "spa_Latn"
text="escucha las palabras de las brujas, los secretos escondidos en la noche, los antiguos dioses invocamos ahora la obra de la magia oculta"
inputs = tokenizer(text="escucha las palabras de las brujas, los secretos escondidos en la noche, los antiguos dioses invocamos ahora la obra de la magia oculta", return_tensors="pt")


outputs = model.generate(inputs['input_ids'], forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"])

print()
print(f"SOURCE TEXT: {text}")
print(f"TRANSLATED TEXT: {tokenizer.batch_decode(outputs, skip_special_tokens=True)}")
print()
