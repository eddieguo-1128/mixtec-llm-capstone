import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from openai import OpenAI 

client = OpenAI(
  api_key="sk-uAAH2jB9Stbfdh_sNSHsiw",
  base_url="https://cmu.litellm.ai"
)

# prepare prompts
messages = [{"role": "system", "content": "You are a helpful assistant that can translate transcriptions of Yoloxóchitl Mixtec language to Spanish."}]
with open("system_prompts_max.jsonl", "r") as file:
    for line in file:
        messages.append(json.loads(line.strip()))

user_prompts = []
with open("user_prompts.jsonl", "r") as file:
    for line in file:
        user_prompts.append(json.loads(line.strip()))

# response generation
responses = []
for prompt in user_prompts:
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages+[prompt],
        temperature=0,
        )
    responses.append(response.choices[0].message.content)

with open("output.txt", "w") as output_file:
    for response in responses:
        output_file.write(response + "\n")  # Write each response on a new line

# Evaluation - BLEU score
references = [
    ["Ya ves que nosotros no comemos cosas crudas sino cosas cocidas al fuego, ahora"],
    ["ojalá que no nos agarre rencor, porque su milpa ya se secó, así me han contado\", dijo su suegra."],
    ["\"Se secó porque así lo quiso él\", dijo el señor de la lluvia."],
    ["Llovía sobre la milpa del señor de la lluvia, pero sobre la milpa del señor del fuego no llovía, así que su milpa se secó."],
    ["Entonces el señor del fuego le dijo a su mujer, \"Ahora, tú,"]
]

candidates = [response.split(" ") for response in responses]
print('Candidates:', candidates)

# Calculate BLEU score across the corpus
bleu_score = corpus_bleu(references, candidates)
print("BLEU score:", bleu_score)
    