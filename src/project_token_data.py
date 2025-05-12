import typer

## Local imports
from utils import load_translator_config, load_aligner_config
from translator_model import TranslatorConfig, BaseTranslatorModel
from alignment_model import AlignerConfig, BaseAlignmentModel


def main(
    config_name: str = 'config.yaml'
):
    
    translator_config = load_translator_config(config_name)
    alignment_config_yaml = load_aligner_config(config_name)

    source = [
        {
            "sentence": "Back in school, I walked to the library every afternoon.",
            "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "sentence": "When we were kids, we played cricket in the streets until sunset.",
            "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O", "O", "O"]
        },
        {
            "sentence": "Every summer, I visited my grandparents in the countryside.",
            "ner_tags": ["O", "B-TIME", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "O"]
        },
        {
            "sentence": "I used to read comic books under the mango tree after lunch.",
            "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O", "B-TIME", "O"]
        }
    ]

    intermediate = []

    translator_config = TranslatorConfig(
        batch_size=int(translator_config.get('batch_size', 4)),
        model_checkpoint_dir=translator_config.get('model_checkpoint_dir', ''),
        q_config=translator_config.get('q_config', '')
    )

    aligner_config = AlignerConfig(
        model=alignment_config_yaml.get('model', 'bert-base-multilingual-cased'),
        output_file=alignment_config_yaml.get('output_file','content/dummty.txt'),
        align_layer=int(alignment_config_yaml.get('align_layer', 8)),
        threshold=float(alignment_config_yaml.get('threshold', 0.3)),
        probs_layer=alignment_config_yaml.get('probs_layer', 'softmax'),
    )

    # # Model init
    translator_model = BaseTranslatorModel(config=translator_config)
    aligner = BaseAlignmentModel(config=aligner_config)

    # Translate a sentence
    for i in range(len(source)):
        source_lang_sentence, ner_tags = [source[i]['sentence']], source[i]['ner_tags']
        print(source_lang_sentence, ner_tags)

        translations = translator_model.batch_translate(source_lang_sentence)
        print(source_lang_sentence, translations)

        for i in range(len(translations)):

            source_sentence, target_sentence = source_lang_sentence[i], translations[i]
            print("Doing something: ", source_sentence, target_sentence)
            
            aligned_words = aligner.get_alignments(source_sentence, target_sentence)
            print(aligned_words)

            print("Aligning translated words with their NER tags")
            intermediate.append({
                'source_sentence': source_sentence[0].split(" "),
                'target_sentence': target_sentence[0].split(" "),
                'source_ner_tags': ner_tags,
                'alignments': aligned_words,
                'target_ner_tags': ["O"] * len(target_sentence.split(" "))
            })

    ## basic version today
    for item in intermediate.items():
        for alignment in item['alignments']:
            print(
                item['source_sentence'][alignment[0]], 
                item['target_sentence'][alignment[1]]
            )
            
            item['target_ner_tags'][ item['target_sentence'][ alignment[1]] ] = item['source_ner_tags'][alignment[0]]


if __name__ == "__main__":
    typer.run(main)