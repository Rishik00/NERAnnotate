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

    source_lang_sentence = [
        "Back in school, I walked to the library every afternoon.",
        "When we were kids, we played cricket in the streets until sunset.",
        "Every summer, I visited my grandparents in the countryside.",
        "I used to read comic books under the mango tree after lunch.",
    ]

    translator_config = TranslatorConfig(
        batch_size=translator_config.get('batch_size', 4),
        model_checkpoint_dir=translator_config.get('model_checkpoint_dir', ''),
        q_config=translator_config.get('q_config', '')
    )

    aligner_config = AlignerConfig(
        model=alignment_config_yaml.get('model', 'bert-base-multilingual-cased'),
        output_file=alignment_config_yaml.get('output_file','content/dummty.txt'),
        align_layer=alignment_config_yaml.get('align_layer', 8),
        threshold=float(alignment_config_yaml.get('threshold', 0.3)),
        probs_layer=alignment_config_yaml.get('probs_layer', 'softmax'),
    )

    # # Model init
    translator_model = BaseTranslatorModel(config=translator_config)
    aligner = BaseAlignmentModel(config=aligner_config)

    # Translate a sentence
    translations = translator_model.batch_translate(source_lang_sentence)
    print(f"Translation: {translations}")

    print(source_lang_sentence, translations)
    for i in range(len(translations)):

        source_sentence, target_sentence = source_lang_sentence[i], translations[i]
        print("Doing something: ", source_sentence, target_sentence)
        
        aligned_words = aligner.get_alignments(source_sentence, target_sentence)
        print(aligned_words)

if __name__ == "__main__":
    typer.run(main)