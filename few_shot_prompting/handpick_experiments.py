import json
import os
import litellm
import random
import tqdm
import os
import pandas as pd
from few_shot_prompting.metrics import calculate_chrf
from select_examples import top_k_examples, read_dict, lexeme_to_gloss_mapping, split_sentence
from nltk.translate import chrf
from dotenv import load_dotenv

def perform_experiment_mix2spa(k, metric, test_data):
    setup = f'{metric}_{k}'
    score = 0
    print(f'Experimenting with the following setup: similarity_metric={metric}, num_shots={k}')
    transcriptions = test_data['cleaned_transcription'].tolist()
    translations = test_data['translation'].tolist()
    for translation, transcription in zip(translations, transcriptions):
        few_shot_examples = top_k_examples(train_data_path, transcription, k, metric, dictionary)
        messages = generate_prompt_mix2spa(transcription, few_shot_examples, False, False)
        response = litellm.completion(
            model='gpt-4o',
            api_key=llm_api_key,
            base_url=llm_base_url,
            messages=messages,
            max_tokens=200,
        )
        model_translation = response.choices[0].message.content
        chrf = calculate_chrf(model_translation, [translation])
        with open("handpick_examples_result_no_dict.txt", "a") as file:
            file.write(f"[Setup]: {setup}\n")
            file.write(f"[Sentence to translate]: {transcription}\n")
            file.write(f"[Reference]: {translation}\n")
            file.write(f"[Model Output]: {model_translation}\n")
            file.write(f"[CHRF]: {chrf}\n")
            file.write(f"[Prompt]:\n{messages[1]['content']}\n\n")
        score += chrf

    print(f'Average CHRF for {setup}: {score / len(translations)}')
    with open("handpick_examples_result_no_dict.txt", "a") as file:
        file.write(f"Average CHRF for {setup}: {score / len(translations)}\n")


def generate_prompt_mix2spa(sentence_to_translate, few_shot_examples, use_dict=True, sample_contain_dict=False):
    system_prompt = "You are a translator between Mixtec and Spanish. Based on the following examples, translate the user's query. Only output the Spanish translation, nothing else.\n\n"
    if sample_contain_dict:
        pass

    example_translations = few_shot_examples['translation'].tolist()
    example_transcriptions = few_shot_examples['cleaned_transcription'].tolist()

    dict_prompt = 'Use the following lexeme dictionary to support your translation:\n'
    lexemes = split_sentence(sentence_to_translate.lower())
    lexemes = list(set(lexemes))  # only keep unique lexemes
    for lexeme in lexemes:
        if lexeme in dictionary:
            # turn dictionary[lexeme] (a list) to a string for LLM
            meaning_str = f'{lexeme}: {", ".join(dictionary[lexeme])}'
            dict_prompt += meaning_str + '\n'

    example_prompt = ''
    for i in range(k):
        example_prompt += f'Example {i + 1}: \n'
        example_prompt += f'Mixtec: {example_transcriptions[i]}\n'
        example_prompt += f'Spanish: {example_translations[i]}\n\n'

    task_prompt = '\nTask (translate the following Mixtec):\n'
    task_prompt += f'Mixtec: {sentence_to_translate}\nSpanish:'

    if use_dict:
        full_prompt = example_prompt + dict_prompt + task_prompt
    else:
        full_prompt = example_prompt + task_prompt

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
        {"role": "user", "content": full_prompt}
    ]

    return messages


if __name__ == "__main__":
    # experiment setup
    metrics = ['chrf', 'lexeme_recall', 'random']
    num_shots = [10, 20, 50]

    # prepare data
    load_dotenv()

    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")

    train_data_path = "../data/train-00000-of-00001.parquet"
    test_data_path = "../data/test-00000-of-00001.parquet"
    dict_path = 'dictionary.json'
    dictionary = read_dict(dict_path)

    df = pd.read_parquet(test_data_path)
    df = df[df['translation'] != '']

    # do a preliminary experiment on 50 samples, remove when run on the whole test set
    df = df.head(50)
    # print(df.shape)

    for k in num_shots:
        for metric in metrics:
            perform_experiment_mix2spa(k, metric, df)







