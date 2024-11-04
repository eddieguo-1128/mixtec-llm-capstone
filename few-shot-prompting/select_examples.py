import pandas as pd
import random
from metrics import calculate_chrf
import re
import json
from collections import defaultdict


def pairwise_similarity(sentence_to_translate, sentence_in_dataset, metric='random', dictionary=None):
    available_metrics = ['random', 'lexeme_recall', 'chrf']
    if metric not in available_metrics:
        metric = 'random'

    if metric == 'random':
        return random.random()

    elif metric == 'lexeme_recall':
        # providing examples which have the same words/morphemes as the target sentence
        splitted_sent1 = split_sentence(sentence_to_translate)
        splitted_sent2 = split_sentence(sentence_in_dataset)

        # exclude lexemes not in the dictionary
        filtered_lexemes_sent1 = [item for item in splitted_sent1 if item in dictionary]
        filtered_lexemes_sent2 = [item for item in splitted_sent2 if item in dictionary]

        unique_lexemes1 = set(filtered_lexemes_sent1)
        unique_lexemes2 = set(filtered_lexemes_sent2)
        intersection = unique_lexemes1.intersection(unique_lexemes2)
        lexeme_recall = len(intersection) / len(unique_lexemes1) if unique_lexemes1 else 0

        # Prevent including the sentence to translate in returned examples.
        if lexeme_recall > 0.99:
            lexeme_recall = 0
        return lexeme_recall

    elif metric == 'chrf':
        chrf_similarity = calculate_chrf(sentence_to_translate, [sentence_in_dataset])

        # Prevent including the sentence to translate in returned examples.
        if chrf_similarity > 99.9:
            chrf_similarity = 0
        return chrf_similarity


def top_k_examples(dataset_path, sentence_to_translate, k, metric='random', dictionary=None):
    df = pd.read_parquet(dataset_path)
    transcriptions = df['cleaned_transcription'].tolist()
    similarities = []
    for transcription in transcriptions:
        similarities.append(pairwise_similarity(sentence_to_translate, transcription, metric, dictionary))
    df['similarity'] = similarities
    sorted_df = df.sort_values(by='similarity', ascending=False)
    return sorted_df.head(k)


def split_sentence(sent):
    sent = sent.replace("(", "").replace(")", "")
    # separate when encounter the following: space, punct, equal sign
    segments = re.split(r'(?<=\w)(=)|[ \.,;!?]', sent)
    segments = [seg for seg in segments if seg]

    # Merge equal signs with the following element
    merged_segments = []
    skip_next = False
    for i, seg in enumerate(segments):
        if skip_next:
            skip_next = False
            continue
        if seg == "=" and i + 1 < len(segments):
            merged_segments.append(seg + segments[i + 1])
            skip_next = True
        else:
            merged_segments.append(seg)

    return merged_segments


def read_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    lexeme_dict = defaultdict(list)
    for entry in json_data:
        lexeme = entry.get("lexeme")
        gloss = entry.get("gloss")
        alternate_forms = entry.get("alternate_forms", [])

        # Add the main lexeme to the dictionary
        if lexeme and gloss:
            lexeme_dict[lexeme].append(gloss)

        # Add each alternate form to the dictionary
        for form in alternate_forms:
            lexeme_dict[form].append(gloss)

        # Convert defaultdict to a regular dictionary for output
    return dict(lexeme_dict)


if __name__ == "__main__":
    example_sentence_to_translate = """tan3 ta1 kum3pa4ri2=ra1 kan4 ndi4, sa3kan4 xa'4an1 ba3 ta1 kan4 ji'4in4=ra2 tan3"""
    example_sentence_in_dataset = """tan3 ko14o(3)=a2 kan4 i4xa3=ra2 ji'4in4 kum3pa4ri2=ra1, tan3 ta1 ko4ndo3 ta1 chi1tun3 ña1a4 ndi4"""
    dataset_path = '../data/train-00000-of-00001.parquet'
    dict_path = 'dictionary.json'
    metric = 'lexeme_recall'
    k = 30
    dictionary = read_dict(dict_path)

    chosen_examples = top_k_examples(dataset_path, example_sentence_to_translate, k, metric, dictionary)
    chosen_examples.to_csv('top_k_examples.csv', index=False)
    print(f'Example sentence to translate: {example_sentence_to_translate}')
    print(f'Found top {k} examples, alongwith translation and similarity scores:')
    print(chosen_examples['cleaned_transcription'].tolist())
    print(chosen_examples['translation'].tolist())
    print(chosen_examples['similarity'].tolist())

    # print(read_dict(dict_path))
    # for element in example_splitted_sent:
    #     if element in dictionary:
    #         print(f'\"{element}\" is found in the dictionary. Gloss: {dictionary[element]}')
    #     else:
    #         print(f'\"{element}\" is not found in the dictionary.')


