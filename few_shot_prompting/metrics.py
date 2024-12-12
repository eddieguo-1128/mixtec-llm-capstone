import sacrebleu
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import BLEU, CHRF


# sacrebleu: 0 to 100, higher score = higher quality
def calculate_sentence_bleu(output, reference):
    return sacrebleu.sentence_bleu(output, reference).score

def calculate_corpus_bleu(output, reference):
    bleu = BLEU()
    return bleu.corpus_score(output, reference)

# chrf: 0 to 100, higher score = higher quality
def calculate_sentence_chrf(output, reference):
    return sacrebleu.sentence_chrf(output, reference).score

def calculate_corpus_chrf(output, reference):
    chrf = CHRF()
    return chrf.corpus_score(output, reference)


# comet: generally -1 to 1 (may outside this range slightly), higher score = higher quality
def calculate_comet(outputs, references, sources, batch_size=8, gpus=0):
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)

    data = [
        {"src": src, "mt": mt, "ref": ref} 
        for src, mt, ref in zip(sources, outputs, references)
    ]

    scores = comet_model.predict(data, batch_size=batch_size, gpus=gpus)
    return scores["system_score"]


def eval_all_metrics(output, reference, source):
    result = {'bleu': calculate_sentence_bleu(output, reference), 'comet': calculate_comet(output, reference, source),
              'chrf': calculate_sentence_chrf(output, reference)}
    return result


if __name__ == "__main__":
    example_sources = [
        """ke13e3=na2 kwa'1an1=na1 mi4i4 ti4 ba42 kwa'1an1 kum3pa4ri2=ra1 ji'4in4=ra3""",
        """tan3 kwa'1an1=na1 sa3kan4 i3in3 i3chi4 kan4 tan3 ka4chi2=ra1 ji'4in4 kum3pa4ri2=ra1 ndi4""",
        """kum3pa4ri2 ndi4, mi4i4 kwa'1an(1)=e4 kan4 ndi4 ba'1a3=ni42 ka4a4 i3in3 ñu3u2 mi4i4 xa1a(1)=e4 kan4 ndi4""",
        """ya1 yo'4o4 ya1 yo'4o4 ba42 i4yo2 kan4 tan3 mi4i4 kwa'1an(1)=e4 ndu3ku(4)=e4 ya1 ko4ndo3 vida ña'1a(3)=e4 ba42 ndi4 i3kan4 i4yo2 xu'14un4 i4yo2"""
    ]
    example_references = [
        ['Así que los dos se fueron de viaje.'],
        ['Se fueron, mientras iban caminando el compadre envidioso le dijo a su compadre,'],
        ['"Compadre, ese pueblo a donde vamos es muy bonito,'],
        ['hay muchas cosas, vamos a conseguir muchas cosas, vamos a buscar nuestra vida, allá hay mucho dinero".']
    ]
    example_outputs = [
        'sabía que el viaje era para acabar con la vida de su compadre',
        'planeó el viaje para hacerle daño a su compadre sin que él lo supiera',
        'su compadre sabía que el viaje era peligroso, pero aún así decidió acompañarlo en el trayecto.',
        'caminaban juntos, caminaban juntos hacia el lugar donde se suponía que ocurriría el final de su vida, sin saber que aquel lugar sería su última morada.'
    ]

    model_path = download_model("Unbabel/wmt20-comet-da")
    comet_model = load_from_checkpoint(model_path)

    for output, ref, src in zip(example_outputs, example_references, example_sources):
        print(eval_all_metrics(output, ref, src))
