import json
import os
import litellm
import random
import tqdm
import os
import pandas as pd
from select_examples import top_k_examples,read_dict,lexeme_to_gloss_mapping
from nltk.translate import chrf
from dotenv import load_dotenv

load_dotenv()

llm_api_key = os.getenv("LLM_API_KEY")
llm_base_url = os.getenv("LLM_BASE_URL")

train_data_path = "../data/train-00000-of-00001.parquet"
test_data_path = "../data/test-00000-of-00001.parquet"
dict_path = 'dictionary.json'
dictionary = read_dict(dict_path)

# Load top-k examples
df = pd.read_parquet(test_data_path)
df = df[df['translation'] != '']
df = df.sample(n=50, random_state=42)
df['lexeme_gloss_mapping'] = df['cleaned_transcription'].apply(lambda x: lexeme_to_gloss_mapping(x, dictionary))

# Create the prompt
def format_lexeme_gloss(lexeme_gloss_mapping):
    description = ""
    for lexeme, glosses in lexeme_gloss_mapping.items():
        gloss_str = ", ".join([gloss for gloss in glosses if gloss is not None])
        description += f'the word "{lexeme}" means "{gloss_str}"; '
    return description.rstrip("; ")


all_outputs = []
for i, x in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
    # system prompt
    metric = 'chrf'
    k = 10
    top_k_examples_df = top_k_examples(train_data_path, x['cleaned_transcription'], k, metric, dictionary)

    system_prompt = "You are a translator between Mixtec and Spanish. Based on the following examples, translate the user's query. Only output the translated text, nothing else.\n\n"
    for _, row in top_k_examples_df.iterrows():
        example_transcription = row['cleaned_transcription']
        example_translation = row['translation']
        example_mapping = row['lexeme_gloss_mapping']
        if isinstance(example_mapping, str):
            example_mapping = eval(example_mapping)
        example_description = format_lexeme_gloss(example_mapping)

        system_prompt += f"Translate the following sentence to Spanish: {example_transcription}\n"
        system_prompt += f"In this context, {example_description}.\n"
        system_prompt += f"The full translation to Spanish is: {example_translation}\n\n"

    # user prompt
    cleaned_transcription = x['cleaned_transcription']
    lexeme_gloss_mapping = x['lexeme_gloss_mapping']
    if isinstance(lexeme_gloss_mapping, str):
        lexeme_gloss_mapping = eval(lexeme_gloss_mapping)
    lexeme_gloss_description = format_lexeme_gloss(lexeme_gloss_mapping)

    user_prompt = f"Now it's your turn!\nTranslate the following sentence to Spanish: {cleaned_transcription}\n"
    user_prompt += f"In this context, {lexeme_gloss_description}.\n"
    user_prompt += f"The full translation to Spanish is:"

    messages = [
        {
            "role": "system",
            "content": [
                {
                  "type": "text",
                  "text": system_prompt,
                  "cache_control": {
                    "type": "ephemeral"
                  }
                }
            ]
         },
        {"role": "user", "content": user_prompt}
    ]
    response = litellm.completion(
        model='gpt-4o',
        api_key=llm_api_key,
        base_url=llm_base_url,
        messages=messages,
        max_tokens=200,
    )
    all_outputs.append(response.choices[0].message.content)

df['generated_translation'] = all_outputs
df.to_csv("result.csv")


# measure chrf between all_outputs and reference translations
chrf_scores = []
for i, x in enumerate(all_outputs):
    generated_translation = x
    reference_translation = df.iloc[i]["translation"]
    score = chrf(generated_translation, reference_translation)
    chrf_scores.append(score)

    print(f"Example {i+1}: {generated_translation}")
    print(f"Reference: {reference_translation}")
    print(f"CHRF score: {score}")
    print("\n")

average_chrf_score = sum(chrf_scores) / len(chrf_scores)
print(f"Average CHRF score: {average_chrf_score}")