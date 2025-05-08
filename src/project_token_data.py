import typer
from typing import List, Dict, Optional
from rich import print as rprint
from dataclasses import dataclass
from datasets import load_dataset

## Local imports
from utils import setup_logging, load_aligner_config, load_translator_config, load_config
from alignment_model import BaseAlignmentModel, AlignerConfig
from translator_model import BaseTranslatorModel, TranslatorConfig

def main(
    config_name: str,
):
    print("Loaded loggers")
    setup_logging('token_ner.log')

    print("Loading configs")

    translator_config_yaml = load_translator_config(config_name)
    alignment_config_yaml = load_aligner_config(config_name)
    overall_config = load_config(config_name)

    translator_config = TranslatorConfig(
        batch_size=translator_config_yaml.get('batch_size', 4),
        model_checkpoint_dir=translator_config_yaml.get('model_checkpoint_dir', 'ai4bharat/indictrans2-en-indic-1B'),
        source_lang=translator_config_yaml.get('source_lang', 'en'),
        target_lang=translator_config_yaml.get('target_lang', 'hi'),
        q_config=translator_config_yaml.get('q_config', 'None'),
        device=translator_config_yaml.get('device', 'cpu')
    )

    aligner_config = AlignerConfig(
        model=alignment_config_yaml.get('model', 'bert-base-multilingual-cased'),
        output_file=alignment_config_yaml.get('output_file','content/dummty.txt'),
        align_layer=alignment_config_yaml.get('align_layer', 8),
        threshold=alignment_config_yaml.get('threshold', 0.3),
        probs_layer=alignment_config_yaml.get('probs_layer', 'softmax'),
        device=alignment_config_yaml.get('device', 'cpu')
    )

    translator = BaseTranslatorModel(config=translator_config)
    aligner = BaseAlignmentModel(config=aligner_config)

    ds = load_dataset("DFKI-SLT/few-nerd", "intra")
    data = ds['train']
    batch_size = translator_config_yaml.batch_size

    for idx in range(0, len(data), batch_size):
        print(f"\n[INFO] Processing batch starting at index: {idx}")

        batch = data[idx:idx + batch_size]

        sentences = [' '.join(example['tokens']) for example in batch]
        ner_tags_batch = [example['ner_tags'] for example in batch]

        print(f"[DEBUG] Batch size: {len(sentences)}")
        print(f"[DEBUG] First sentence in batch: {sentences[0]}")

        translations: List[str] = translator.batch_translate(sentences)
        print(f"[DEBUG] Translations: {translations[:1]}")  # show only first for brevity

        alignments = aligner.get_batch_alignments(sentences, translations)
        print(f"[DEBUG] First alignment in batch: {alignments[0]}")

        break  # Dev testing

        # TODO: Reorder NER tags based on alignments
        # TODO: Save outputs to JSON
            

    else:
        pass

if __name__ == "__main__":
    typer.run(main)