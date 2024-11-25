from few_shot_prompting.select_examples import split_sentence, read_dict
from get_dictionary_coverage import main, lookup_stem
import collections
from parse_toolbox import parse_toolbox
import json
import logging
import re
import pandas as pd

from utils import (
    dictionary_fields,
    get_dictionary_entries,
    parse_toolbox,
)


def create_vocab(dictionary_path, vocab_path):
    dict_stems = collections.defaultdict(set)
    entries = parse_toolbox(dictionary_path)
    for entry in entries:
        entry_dict = collections.defaultdict(list)
        for key, value in entry:
            entry_dict[key].append(value.lower())
        lx = entry_dict["lx"][0]
        uid = entry_dict["ref"][0]
        forms = [
            [(f, key) for f in get_dictionary_entries(entry_dict, key)]
            for key in dictionary_fields
        ]
        forms = sum(forms, [])
        for form, key in forms:
            dict_stems[form].add((uid, key))
        dict_stems[lx].add((uid, "lx"))

    with open(vocab_path) as f:
        vocab = json.load(f)

    stems = collections.defaultdict(int)
    for word in vocab:
        stem_brac = re.match(r"[\w'{}>()-]+", word)
        if stem_brac is None:
            logging.warning(f"Cannot find stem for {word}")
            continue
        stem_brac = stem_brac.group()
        stem = stem_brac.translate(str.maketrans("", "", "()"))
        stems[stem] += sum(vocab[word].values())

    no_seg_stems = {stem.replace("-", ""): stem for stem in dict_stems}

    return dict_stems, no_seg_stems


def lookup(lexeme, dict_stems, no_seg_stems, dict_json):
    punct = """!"#$%&'()*+,-./:;<>?@[\]^_`{|}~¿"""
    # starts/ends with punct -> remove that
    while lexeme[0] in punct:
        lexeme = lexeme[1:]
    while lexeme[-1] in punct:
        lexeme = lexeme[:-1]
    result = lookup_stem(lexeme, dict_stems, no_seg_stems)

    return retrieve_meaning(result, dict_json)


def retrieve_meaning(lexeme, dict_json):
    cols = {
        "lx_alt": "alternate_forms",
        "lx_hab": "habitual",
        "lx_comni": "completive",
        "lx_comto": "completive",
        "lx_est": "stative",
        "lx_pro": "progressive",
        "lx_pot": "potential",
        "lx_neg": "negative",
        "lx_imp": "imperative",
        "lx_neg_hab": "negative habitual",
    }
    gloss = []
    sigs = []
    found_field = []

    for entry in dict_json:
        if entry.get("lexeme") == lexeme:
            gloss.append(entry.get("gloss", ""))
            sigs.append(entry.get("sig", []))
            found_field.append('lexeme')
            continue

        for field in cols.values():
            if lexeme in entry.get(field, []):
                gloss.append(entry.get("gloss", ""))
                sigs.append(entry.get("sig", []))
                found_field.append(field)
                break
    return gloss, sigs, found_field


if __name__ == "__main__":
    dictionary_path = 'Active_Yolo-Mixtec_2024-11-09.txt'
    vocab_path = "all_vocab.json"

    dict_stems, no_seg_stems = create_vocab(dictionary_path, vocab_path)
    print(dict_stems)
    train_data_path = "../data/train-00000-of-00001.parquet"
    test_data_path = "../data/test-00000-of-00001.parquet"

    dict_path = 'dictionary.json'
    with open(dict_path, 'r', encoding='utf-8') as file:
        dict_json = json.load(file)

    df = pd.read_parquet(test_data_path)
    df = df[df['translation'] != '']
    transcriptions = df['cleaned_transcription'].tolist()

    unique_lexemes = []
    found_lexemes = []
    miss_lexemes = []
    lexeme_count = 0
    found_count = 0

    for transcription in transcriptions:
        lexemes = split_sentence(transcription.lower())
        punct = """!"#$%&'()*+,-./:;<>?@[\]^_`{|}~¿"""

        for lexeme in lexemes:
            # not a Mixtec word, ignore it
            if not any(char.isdigit() for char in lexeme):
                continue

            # starts/ends with punct -> remove that
            while lexeme[0] in punct:
                lexeme = lexeme[1:]
            while lexeme[-1] in punct:
                lexeme = lexeme[:-1]

            lexeme_count += 1
            if lexeme not in unique_lexemes:
                unique_lexemes.append(lexeme)
            result = lookup_stem(lexeme, dict_stems, no_seg_stems)
            meaning = lookup(lexeme, dict_stems, no_seg_stems, dict_json)

            # record found lexemes
            if result is not None and result not in found_lexemes:
                found_lexemes.append(result)
            if result is not None:
                found_count += 1

            # show cases that founded is not the same as the word in the corpus
            if (result is not None) and lexeme.replace('-', '') != result.replace('-', ''):
                print(f'lookup for {lexeme}, found {result} in the dict')
                print(f'found entry (gloss, sigs, field): {meaning}')
                print('-' * 50)

            if (result is None) and (lexeme not in miss_lexemes):
                miss_lexemes.append(lexeme)

    print(f'Total number of unique lexemes: {len(unique_lexemes)}')
    print(f'Total number of unique found lexemes: {len(found_lexemes)}')
    print(f'Total number of lexemes: {lexeme_count}')
    print(f'Total number of found lexemes: {found_count}')
    print(miss_lexemes)


# test old lookup
# old_lexeme_count = 0
# old_unique_lexemes = []
# old_found_count = 0
# old_unique_founds = []
#
# dict_path = 'dictionary.json'
# dictionary = read_dict(dict_path)
# for transcription in transcriptions:
#     lexemes = split_sentence(transcription.lower())
#     for lexeme in lexemes:
#         # not a Mixtec word, ignore it
#         if not any(char.isdigit() for char in lexeme):
#             continue
#
#         # starts/ends with punct -> remove that
#         punct = """!"#$%&'()*+,-./:;<>?@[\]^_`{|}~¿"""
#         while lexeme[0] in punct:
#             lexeme = lexeme[1:]
#         while lexeme[-1] in punct:
#             lexeme = lexeme[:-1]
#
#         old_lexeme_count += 1
#         if lexeme not in old_unique_lexemes:
#             old_unique_lexemes.append(lexeme)
#
#         if lexeme in dictionary:
#             old_found_count += 1
#             if lexeme not in old_unique_founds:
#                 old_unique_founds.append(lexeme)
#
# print(f'Total number of unique lexemes (old method): {len(old_unique_lexemes)}')
# print(f'Total number of unique found lexemes (old method): {len(old_unique_founds)}')
# print(f'Total number of lexemes (old method): {old_lexeme_count}')
# print(f'Total number of found lexemes (old method): {old_found_count}')

