import json
import os
import litellm
import random
import tqdm
import os
import pandas as pd
from nltk.translate import chrf
from dotenv import load_dotenv

load_dotenv()

llm_model = os.getenv("LLM_MODEL", "openai/neulab/claude-3.5-sonnet-20241022")
llm_api_key = os.getenv("LLM_API_KEY")
llm_base_url = os.getenv("LLM_BASE_URL")

# Load top-k examples
df = pd.read_csv("top_k_examples.csv")
system_prompt_df = df.iloc[:25].reset_index(drop=True)
user_prompt_df = df.iloc[25:30].reset_index(drop=True)

# Concatenate "cleaned_transcription" and "translation"
concatenated_examples = "----------\n".join(
    [f'{row["cleaned_transcription"]}\n{row["translation"]}\n\n' for _, row in system_prompt_df.iterrows()]
)

# Create the prompt
prompt = f"""You are a translator between Mixtec and Spanish. Based on the following examples, translate the user's query. Only output the translated text, nothing else.\n{concatenated_examples}"""

all_outputs = []
for i, x in tqdm.tqdm(user_prompt_df.iterrows(), total=user_prompt_df.shape[0]):
    if i < len(all_outputs):
        continue
    messages = [
        {
            "role": "system",
            "content": [
                {
                  "type": "text",
                  "text": prompt,
                  "cache_control": {
                    "type": "ephemeral"
                  }
                }
            ]
         },
        {"role": "user", "content": x["cleaned_transcription"]}
    ]
    response = litellm.completion(
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
        messages=messages,
        max_tokens=200,
    )
    all_outputs.append(response.choices[0].message.content)



# measure chrf between all_outputs and reference translations
chrf_scores = []
for i, x in enumerate(all_outputs):
    generated_translation = x
    reference_translation = user_prompt_df.iloc[i]["translation"]
    score = chrf(generated_translation, reference_translation)
    chrf_scores.append(score)

    print(f"Example {i+1}: {generated_translation}")
    print(f"Reference: {reference_translation}")
    print(f"CHRF score: {score}")
    print("\n")

average_chrf_score = sum(chrf_scores) / len(chrf_scores)
print(f"Average CHRF score: {average_chrf_score}")