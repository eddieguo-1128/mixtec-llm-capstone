import collections
import logging
import re
import subprocess
import unicodedata

import Levenshtein
import pympi

# from wav2gloss.utils.elan_utils import align_tiers, convert_to_realtime

punctuation = [".", ",", "¡", "!", "¿", "?", '"', ";", ":"]

dictionary_fields = [
    "lx_alt",
    "lx_hab",
    "lx_comni",
    "lx_comto",
    "lx_est",
    "lx_pro",
    "lx_pot",
    "lx_neg",
    "lx_imp",
    "lx_neg_hab",
]


def normalize_text(text, comma=False):
    text = unicodedata.normalize("NFKC", text)
    if not comma:
        return text
    else:
        text = text.replace(",", ", ")
        text = " ".join(text.split())
        return text


def _hard_normalize(text):
    return "".join(
        [
            c
            for c in unicodedata.normalize("NFD", text.lower())
            if unicodedata.category(c) != "Mn"
        ]
    )


def process_file(ipt):
    doc = pympi.Elan.Eaf(ipt)
    tiers_by_speakers = collections.defaultdict(dict)
    for tiername, tier in doc.tiers.items():
        spk = tier[2].get("PARTICIPANT", "UNKNOWN")
        if spk in ["Comments", "Mode", "Fidelity"]:
            continue
        if len(spk) == 0:
            spk = "UNKNOWN"
        # spk_initials = "".join([x[0] for x in spk.split()])
        if tiername == "ASR Hypothesis":
            tiers_by_speakers[spk]["asr"] = tiername
        elif tiername == f"{spk}-ASR":
            tiers_by_speakers[spk]["asr"] = tiername
        elif "surface" in tiername.lower():
            tiers_by_speakers[spk]["surface"] = tiername
        elif "traduccion" in _hard_normalize(tiername):
            tiers_by_speakers[spk]["translation"] = tiername
        elif tiername == spk:
            tiers_by_speakers[spk]["ortho"] = tiername
        elif "correccion" in _hard_normalize(tiername):
            if "ortho_correction" in tiers_by_speakers[spk]:
                logging.warning(f" Orthography collision: {tiername} for file: {ipt}")
            tiers_by_speakers[spk]["ortho_correction"] = tiername
        else:
            logging.info(f" Unknonw tier name: {tiername} for file: {ipt}")
    return doc, tiers_by_speakers


def process_generated_file(ipt):
    doc = pympi.Elan.Eaf(ipt)
    tiers_by_speakers = collections.defaultdict(dict)
    for tiername, tier in doc.tiers.items():
        spk = tier[2].get("PARTICIPANT", "UNKNOWN")
        if spk == "Comments":
            continue
        initials = "".join([x[0] for x in spk.split()]).upper()
        if tiername == spk:
            tiers_by_speakers[spk]["ortho"] = tiername
        elif tiername == f"{initials}-SEG-Corrected":
            tiers_by_speakers[spk]["ortho_seg"] = tiername
        elif tiername == f"{initials}-SURFACE":
            tiers_by_speakers[spk]["surface"] = tiername
        elif tiername == f"{initials}-SURFACE-SEG-Corrected":
            tiers_by_speakers[spk]["surface_seg"] = tiername
        elif tiername == f"{initials}-Translation":
            tiers_by_speakers[spk]["translation"] = tiername
        else:
            logging.info(f" Unknonw tier name: {tiername} for file: {ipt}")
    return doc, tiers_by_speakers


def call_flookup(foma_file, word):
    command = ["flookup", "-ix", foma_file]
    out = subprocess.run(
        command,
        input=word,
        encoding="utf-8",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return out.stdout.strip()


def match_brackets(observed, ortho_seg, markers=["[", "]", "(", ")"]):
    """
    Use edit distance to segment the the observed orthography
    """

    replace_char = 0x1F600
    rep_back = dict()
    for match in re.finditer(r"\{[^\}]*\}", ortho_seg):
        c = chr(replace_char)
        rep_back[c] = match.group(0)
        ortho_seg = ortho_seg.replace(match.group(0), c, 1)
        replace_char += 1

    alledits = collections.defaultdict(list)
    edits = Levenshtein.editops(observed, ortho_seg)

    for o, s, d in edits:
        alledits[s].append((d, o))

    cuts = collections.defaultdict(list)
    for i, c in enumerate(observed):
        if c in markers:
            for d, o in alledits[i]:
                match o:
                    case "insert":
                        continue
                    case "replace":
                        cuts[d].append(c)
                        break
                    case "delete":
                        cuts[d].append(c)
                        break
    cuts = {p: "".join(cuts[p]) for p in cuts}
    cut_points = list(sorted(cuts.keys())) + [len(ortho_seg)]
    observed_seg = ortho_seg[: cut_points[0]]
    for i, pos in enumerate(cut_points[:-1]):
        observed_seg = observed_seg + cuts[pos] + ortho_seg[pos : cut_points[i + 1]]

    # post process
    for match in re.finditer(r"\([^\)]*\)", observed_seg):
        match_str = match.group(0)
        if len(match_str) > 3 and match_str[1] in rep_back:
            adjusted = f"{match_str[1]}({match_str[2:-1]})"
            observed_seg = observed_seg.replace(match_str, adjusted)

    for c, bak in rep_back.items():
        observed_seg = observed_seg.replace(c, bak)

    return observed_seg


# def get_sentences(elan_file, copy_translations=False):
#     doc, tiers_by_speakers = process_file(elan_file)
#     texts = []
#     for spk in tiers_by_speakers:
#         curr_doc = tiers_by_speakers[spk]
#
#         if "ortho_correction" in curr_doc:
#             ortho_tier = curr_doc["ortho_correction"]
#         elif "ortho" in curr_doc:
#             ortho_tier = curr_doc["ortho"]
#         else:
#             logging.error(
#                 " Practical orthography tier not found"
#                 f" for speaker {spk} in file {elan_file}"
#             )
#             continue
#
#         aligned = None
#         if copy_translations and "translation" in curr_doc:
#             aligned, dropped = align_tiers(doc, ortho_tier, curr_doc["translation"])
#             if dropped > 0:
#                 logging.info(
#                     f"Alignment error in {elan_file} for speaker {spk},"
#                     " skipping translation."
#                 )
#                 aligned = None
#
#         if aligned is not None:
#             for val in aligned.values():
#                 text = val["ref"][2]
#                 st = val["ref"][0]
#                 et = val["ref"][1]
#                 transl = val["aligned"][0][3]
#                 texts.append((st, et, spk, normalize_text(text), transl))
#         else:
#             for _, st, et, text in convert_to_realtime(doc, ortho_tier):
#                 texts.append((st, et, spk, normalize_text(text), ""))
#     texts = sorted(texts, key=lambda x: x[0])
#     return texts


def extract_mixtec_tokens(raw_orthography):
    mixtec_tokens = []
    text = normalize_text(raw_orthography, comma=True)
    text = re.sub(r"[\[\]]", "", text.lower())
    text = re.sub(r"\*[^\*]+\*\*", "🚫", text)
    text = re.sub(r"<i>[^<]+</i>", "🚫", text)
    for word in text.split():
        if "..." in word or "🚫" in word:
            continue
        word = word.translate(str.maketrans("", "", "".join(punctuation)))
        if re.search(r"[1-4]", word) and not word.isnumeric():
            mixtec_tokens.append(word)
    return mixtec_tokens


def parse_toolbox(toolbox_file):
    with open(toolbox_file) as f:
        toolbox = f.read()

    # Remove comments
    toolbox = re.sub(r"\\_.*\n", "", toolbox).strip()

    # Split into entries
    entries = re.split(r"\\lx +(.*)\n", toolbox)

    # Parse entries
    parsed_entries = []
    for j in range(1, len(entries), 2):
        lx = entries[j]
        entry = entries[j + 1]
        splitted = re.split(r"\\([\w_]+)[ \n]", entry)
        curr = [("lx", lx)]
        for i in range(1, len(splitted), 2):
            curr.append((splitted[i], splitted[i + 1].strip()))
        parsed_entries.append(curr)
    return parsed_entries


def get_dictionary_entries(entry, key):
    vals = entry.get(key, [])
    vals = [re.sub("[;,]", " ", val).split() for val in vals]
    return sum(vals, [])
