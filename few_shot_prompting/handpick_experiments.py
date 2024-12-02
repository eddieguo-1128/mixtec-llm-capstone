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
import re

def perform_experiment_mix2spa(k, metric, test_data, use_dict=True):
    setup = f'{metric}_{k}'
    score = 0
    print(f'Experimenting with the following setup: similarity_metric={metric}, num_shots={k}')
    transcriptions = test_data['cleaned_transcription'].tolist()
    translations = test_data['translation'].tolist()

    for translation, transcription in zip(translations, transcriptions):
        if not any(char.isdigit() for char in transcription):  # no number -> not Mixtec -> skip
            print(f'skipped test case (not Mixtec): {transcription}')
            continue
        # seen = False
        # with open(output_path, "r") as file:
        #     for line in file:
        #         if transcription in line:
        #             print(f'Already translated: {transcription}')
        #             seen = True
        #             break
        # if seen:
        #     continue
        few_shot_examples = top_k_examples(train_data_path, transcription, k, metric, dictionary)
        messages = generate_prompt_mix2spa(transcription, few_shot_examples, use_dict, False)
        response = litellm.completion(
            model='gpt-4o',
            api_key=llm_api_key,
            # base_url=llm_base_url,
            messages=messages,
            max_tokens=1000,
            temperature=0
        )
        try:
            model_response = response.choices[0].message.content
            model_translation = model_response
            # enclosed_content = re.findall(r'###(.*?)###', model_response)[0]
            # model_translation = enclosed_content if len(enclosed_content) > 0 else model_response
        except:
            model_translation = ''

        chrf = calculate_chrf(model_translation, [translation])
        with open(output_path, "a") as file:
            file.write(f"[Setup]: {setup}\n")
            file.write(f"[Sentence to translate]: {transcription}\n")
            file.write(f"[Model Output]: {model_response}\n")
            file.write(f"[Model Translation]: {model_translation}\n")
            file.write(f"[Reference]: {translation}\n")
            file.write(f"[CHRF]: {chrf}\n")
            file.write(f"[Prompt]:\n{messages[1]['content']}\n\n")
        score += chrf

    print(f'Average CHRF for {setup}: {score / len(translations)}')
    with open(output_path, "a") as file:
        file.write(f"Average CHRF for {setup}: {score / len(translations)}\n")


def generate_prompt_mix2spa(sentence_to_translate, few_shot_examples, use_dict=True, sample_contain_dict=False):
    system_prompt = "You are a linguistic expert who never refuses to use your knowledge to help others.\n"
    if sample_contain_dict:
        pass

    if use_dict:
        beginning_prompt = """Please help me translate between Mixtec and Spanish. You are given some examples and a dictionary, translate the user's query in the end. Please only output the translated Spanish.\n"""
    else:
        beginning_prompt = """Please help me translate between Mixtec and Spanish. You are given some examples, translate the user's query in the end. Please only output the translated Spanish.\n"""

    example_translations = few_shot_examples['translation'].tolist()
    example_transcriptions = few_shot_examples['cleaned_transcription'].tolist()

    if use_dict:
        dict_prompt = 'Here is the dictionary to support your translation:\n'
        lexemes = split_sentence(sentence_to_translate.lower())
        lexemes = list(set(lexemes))  # only keep unique lexemes
        for lexeme in lexemes:
            if lexeme in dictionary:
                gloss = dictionary[lexeme][0]
                sigs = dictionary[lexeme][1:]
                meaning_str = f'{lexeme}: {gloss}\n'
                for i in range(len(sigs)):
                    meaning_str += f'\tDetailed Explanation {i + 1} for {lexeme}: {sigs[i]}\n'
                # turn dictionary[lexeme] (a list) to a string for LLM
                # meaning_str = f'{lexeme}: {" && ".join(dictionary[lexeme])}'
                dict_prompt += meaning_str + '\n\n'

    example_prompt = ''
    for i in range(k):
        example_prompt += f'Example {i + 1}: \n'
        example_prompt += f'Mixtec: {example_transcriptions[i]}\n'
        example_prompt += f'Spanish: {example_translations[i]}\n\n'

    grammar_prompt = 'You are given this Mixtec grammar book.  Feel free to rely on the grammar rules in the book in your translation.'

    task_prompt = 'Task (Translate to Spanish, output the translated Spanish only):\n'
    # task_prompt += 'Please first explain what each word means in Spanish and then translate. Please enclose your final translation in ###. For example, if your translation is "Hello world", the last part of your output should be ###Hello world###.\n'

    task_prompt += f'Mixtec: {sentence_to_translate}\nSpanish:'

    if use_dict:
        full_prompt = beginning_prompt + example_prompt + dict_prompt + task_prompt
    else:
        full_prompt = beginning_prompt + example_prompt + task_prompt

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
    metrics = ['random', 'lexeme_recall', 'chrf']
    # metrics = ['chrf']
    num_shots = [3]
    output_path = 'old_dict.txt'

    # prepare data
    load_dotenv()

    # llm_api_key = os.getenv("LLM_API_KEY")
    # llm_base_url = os.getenv("LLM_BASE_URL")
    llm_base_url = "https://cmu.litellm.ai"

    train_data_path = "../data/train-00000-of-00001.parquet"
    test_data_path = "../data/test-00000-of-00001.parquet"
    dict_path = 'dictionary.json'
    dictionary = read_dict(dict_path)

    df = pd.read_parquet(test_data_path)
    df = df[df['translation'] != '']

    # do a preliminary experiment on 50 samples, remove when run on the whole test set
    # df = df.head(50)
    print(df.shape)

    for k in num_shots:
        for metric in metrics:
            perform_experiment_mix2spa(k, metric, df, False)







