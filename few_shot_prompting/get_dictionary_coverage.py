import collections
from datetime import datetime
import json
import logging
import re

import pandas as pd

from utils import (
    dictionary_fields,
    get_dictionary_entries,
    parse_toolbox,
)

rm_segs = str.maketrans("", "", "-")


def _lookup_form(stem, vocab, vocab_nosegs):
    if stem in vocab:
        return stem
    else:
        stem_nosegs = stem.translate(rm_segs)
        if stem_nosegs in vocab_nosegs:
            return vocab_nosegs[stem_nosegs]
    return None


def lookup_stem(stem, vocab, vocab_nosegs):
    found = _lookup_form(stem, vocab, vocab_nosegs)
    if found:
        return found
    forms = []
    curr_stem = stem

    # ni- completives; 1 tone is base form, 14 is negative, 4 is an alternate form
    compl = re.match(r"ni(?:14|4|1)-", curr_stem)
    if compl:
        root = curr_stem[compl.end() :]
        forms.extend(["ni1-" + root, root])
        curr_stem = root

    # nd- iterative; 3 tone is base form
    iterative = re.match(r"(nd[aeiou])[1-4]+-?", curr_stem)
    if iterative:
        root = curr_stem[len(iterative.group()) :]
        prefix = iterative.group(1)
        forms = [f"{prefix}3-{root}", root]
        curr_stem = root

    # x/s/j alternations of the first consonant
    if curr_stem[0] in "xsj":
        rest = curr_stem[1:]
        forms.extend([f"x{rest}", f"s{rest}", f"j{rest}"])

    for form in forms:
        found = _lookup_form(form, vocab, vocab_nosegs)
        if found:
            return found
    return None


def main(dictionary, vocab="all_vocab.json", results=False):

    dict_stems = collections.defaultdict(set)
    entries = parse_toolbox(dictionary)
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

    with open(vocab) as f:
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

    print("Stems in dictionary: ", len(dict_stems))
    print("Stems in corpus: ", len(stems))
    all_toks = sum(stems.values())
    print("Tokens in corpus: ", all_toks)

    accounted = []
    status = []
    no_seg_stems = {stem.replace("-", ""): stem for stem in dict_stems}
    for stem in stems:
        found = lookup_stem(stem, dict_stems, no_seg_stems)
        if found is not None:
            accounted.append(stem)
            entries = list(dict_stems[found])
            field = entries[0][1]
            entries = [e[0] for e in entries]
        else:
            found = ""
            entries = []
            field = ""
        status.append(
            {"observed": stem, "found": found, "entries": len(entries), "field": field}
        )
    print(
        f"Accounted stems: {len(accounted)}, {len(accounted) / len(stems) * 100}%",
    )
    print("Unkown stems: ", len(stems) - len(accounted))

    accounted_toks = sum([stems[s] for s in accounted])
    print(f"Accounted tokens: {accounted_toks}, {accounted_toks / all_toks * 100}%")

    ambiguous = []
    for stem in accounted:
        if len(dict_stems[stem]) > 1:
            ambiguous.append(stem)
    print("Ambiguous stems: ", len(ambiguous))
    ambig_toks = sum([stems[s] for s in ambiguous])
    print(f"Ambiguous tokens: {ambig_toks}, {ambig_toks / all_toks * 100}%")

    if results:
        summary = pd.DataFrame(status)
        summary.to_excel(
            "stem_vocab_coverage_{}.xlsx".format(
                datetime.today().strftime(r"%Y-%m-%d")
            ),
            index=False,
        )

    return pd.DataFrame(status)


if __name__ == "__main__":
    dictionary_path = 'Active_Yolo-Mixtec_2024-11-09.txt'
    main(dictionary_path, results=True)
