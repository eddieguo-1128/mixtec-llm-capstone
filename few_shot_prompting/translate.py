import json
import os
import litellm
import os
import pandas as pd
from metrics import calculate_sentence_chrf, calculate_sentence_bleu, calculate_corpus_bleu, calculate_corpus_chrf, calculate_comet
from select_examples import top_k_examples, read_dict, lexeme_to_gloss_mapping, split_sentence
from dotenv import load_dotenv
from tqdm import tqdm
from generate_dict_prompt import generate_dict_prompt, create_vocab, lookup
from get_dictionary_coverage import main, lookup_stem

def generate_grammar_prompt(grammar_file="grammar/grammar.md"):
    with open(grammar_file, "r") as file:
        grammar_rules = file.read()
    return f"Here are some grammar rules to help with translation:\n\n{grammar_rules}\n\n"

def generate_prompt(sentence_to_translate, few_shot_examples, dict_stems, no_seg_stems, dict_json, experiment):
    system_prompt = "You are a linguistic expert who never refuses to use your knowledge to help others.\n"

    beginning_prompt = """Please help me translate between Mixtec and Spanish. You are given some examples and a dictionary, translate the user's query in the end. Please only output the translated Spanish.\n"""

    example_translations = few_shot_examples['translation'].tolist()
    example_transcriptions = few_shot_examples['cleaned_transcription'].tolist()

    example_prompt = ''
    k = len(few_shot_examples)
    for i in range(k):
        example_prompt += f'Example {i + 1}: \n'
        example_prompt += f'Mixtec: {example_transcriptions[i]}\n'
        example_prompt += f'Spanish: {example_translations[i]}\n\n'

    dict_prompt = 'Here is the dictionary to support your translation:\n'
    lexemes = split_sentence(sentence_to_translate.lower())
    lexemes = list(set(lexemes))  # only keep unique lexemes
    for lexeme in lexemes:
        result = lookup_stem(lexeme, dict_stems, no_seg_stems)
        meaning = lookup(result, dict_stems, no_seg_stems, dict_json)

        if meaning is None:
            # not in the dictionary, skip
            continue
        gloss, sigs, field, original_word = meaning
        for i in range(len(gloss)):
            dict_prompt += generate_dict_prompt(lexeme, result, gloss[i], sigs[i], field[i], original_word[i])
        dict_prompt += '\n'

    task_prompt = 'Task (Translate to Spanish, output the translated Spanish only):\n'
    task_prompt += f'Mixtec: {sentence_to_translate}\nSpanish:'

    grammar_prompt = generate_grammar_prompt(grammar_file="grammar/grammar.md")

    if experiment == 'baseline':
        full_prompt = beginning_prompt + example_prompt + task_prompt
    elif experiment == 'dict':
        full_prompt = beginning_prompt + example_prompt + dict_prompt + task_prompt
    elif experiment == 'grammar':
        full_prompt = beginning_prompt + example_prompt + grammar_prompt + task_prompt
    elif experiment == 'dict_grammar':
        full_prompt = beginning_prompt + example_prompt + dict_prompt + grammar_prompt + task_prompt
    
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


def perform_experiment_updated_dict(k, metric, test_data, dict_stems, no_seg_stems, dict_json, experiment):
    setup = f'{metric}_{k}'
    model_translations = []
    references = []
    chrf_scores, bleu_scores = 0, 0
    print(f'Experimenting with the following setup: similarity_metric={metric}, num_shots={k}')
    transcriptions = test_data['cleaned_transcription'].tolist()
    translations = test_data['translation'].tolist()
    with tqdm(total=len(transcriptions), desc="Processing Translations", unit="sentence") as pbar:
        for translation, transcription in zip(translations, transcriptions):
            if not any(char.isdigit() for char in transcription):  # no number -> not Mixtec -> skip
                print(f'skipped test case (not Mixtec): {transcription}')
                continue
            few_shot_examples = top_k_examples(train_data_path, transcription, k, metric, dictionary)
            messages = generate_prompt(transcription, few_shot_examples, dict_stems, no_seg_stems, dict_json, experiment)
            print(messages)
            response = litellm.completion(
                model='openai/gpt-4o',
                api_key=llm_api_key,
                #base_url=llm_base_url,
                messages=messages,
                max_tokens=1000,
                temperature=0
            )
            model_response = response.choices[0].message.content

            model_translations.append(model_response)
            references.append([translation])

            chrf = calculate_sentence_chrf(model_response, [translation])
            bleu = calculate_sentence_bleu(model_response, [translation])

            chrf_scores += chrf
            bleu_scores += bleu

            with open(output_path, "a") as file:
                file.write(f"[Setup]: {setup}\n")
                file.write(f"[Sentence to translate]: {transcription}\n")
                file.write(f"[Model Output]: {model_response}\n")
                file.write(f"[Reference]: {translation}\n")
                file.write(f"[CHRF]: {chrf}\n")
                file.write(f"[BLEU]: {bleu}\n")
                file.write(f"[Prompt]:\n{messages[1]['content']}\n\n")
            pbar.update(1)
    
    corpus_chrf = calculate_corpus_chrf(model_translations, references)
    corpus_bleu = calculate_corpus_bleu(model_translations, references)
    comet = calculate_comet(model_translations, references, transcriptions)
    average_chrf = chrf_scores / len(transcriptions)
    average_bleu = bleu_scores / len(transcriptions)

    with open(output_path, "a") as file:
        file.write(f"Corpus-Level CHRF for {setup}: {corpus_chrf}\n")
        file.write(f"Corpus-Level BLEU for {setup}: {corpus_bleu}\n")
        file.write(f"COMET score for {setup}: {comet}\n")
        file.write(f"Average CHRF for {setup}: {average_chrf}\n")
        file.write(f"Average BLEU for {setup}: {average_bleu}\n")


if __name__ == "__main__":
    train_data_path = "../data/train-00000-of-00001.parquet"
    test_data_path = "../data/test-00000-of-00001.parquet"
    dict_path = 'dictionary/dictionary.json'
    with open(dict_path, 'r', encoding='utf-8') as file:
        dict_json = json.load(file)
    dictionary = read_dict(dict_path)

    df = pd.read_parquet(test_data_path)
    df = df[df['translation'] != '']

    # do a preliminary experiment on 50 samples, remove when run on the whole test set
    df = df.head(3)
    print(f'Experimenting on {len(df)} test sentences.')

    test_data = df
    transcriptions = test_data['cleaned_transcription'].tolist()
    translations = test_data['translation'].tolist()

    dictionary_path = 'dictionary/Active_Yolo-Mixtec_2024-11-09.txt'
    vocab_path = "dictionary/all_vocab.json"
    dict_stems, no_seg_stems = create_vocab(dictionary_path, vocab_path)

    load_dotenv()
    llm_api_key = os.getenv("LLM_API_KEY")
    #llm_api_key = 'sk-zT5PxQL7dYirU7FgIzE6dA'
    #llm_base_url = "https://cmu.litellm.ai"

    metric = 'chrf'
    num_shots = 3
    experiments = ['baseline','dict','grammar','dict_grammar']

    for experiment in experiments:
        output_path = f'output_{experiment}.txt'
        perform_experiment_updated_dict(num_shots, metric, df, dict_stems, no_seg_stems, dict_json, experiment)




