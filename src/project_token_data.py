import typer
import json
import os

# Local imports
from utils import load_translator_config, load_aligner_config, clear_memory
from translator_model import TranslatorConfig, BaseTranslatorModel
from alignment_model import AlignerConfig, BaseAlignmentModel


def main(
    config_name: str = 'config.yaml',
    output_file: str = 'output_file.json'
):
    # Load configs
    translator_config_yaml = load_translator_config(config_name)
    alignment_config_yaml = load_aligner_config(config_name)

    # Example sentences
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

    # Rebuild config dataclasses
    translator_config = TranslatorConfig(
        batch_size=int(translator_config_yaml.get('batch_size', 4)),
        model_checkpoint_dir=translator_config_yaml.get('model_checkpoint_dir', ''),
        q_config=translator_config_yaml.get('q_config', '')
    )

    aligner_config = AlignerConfig(
        model=alignment_config_yaml.get('model', 'bert-base-multilingual-cased'),
        output_file=alignment_config_yaml.get('output_file', 'content/dummy.txt'),
        align_layer=int(alignment_config_yaml.get('align_layer', 8)),
        threshold=float(alignment_config_yaml.get('threshold', 0.3)),
        probs_layer=alignment_config_yaml.get('probs_layer', 'softmax'),
    )

    # Initialize models
    translator_model = BaseTranslatorModel(config=translator_config)
    aligner = BaseAlignmentModel(config=aligner_config)

    intermediate = []

    # Translate + align
    for i in range(len(source)):
        source_lang_sentence = [source[i]['sentence']]
        ner_tags = source[i]['ner_tags']
        print(f"[INFO] Translating: {source_lang_sentence[0]}")

        translations = translator_model.batch_translate(source_lang_sentence)
        target_sentence = translations[0]

        # Get alignments
        aligned_words = aligner.get_alignments(source_lang_sentence[0], target_sentence)
        print("[INFO] Alignments:", aligned_words)

        # Initialize tag transfer
        source_tokens = source_lang_sentence[0].split()
        target_tokens = target_sentence.split()
        target_ner_tags = ["O"] * len(target_tokens)
        english_to_hindi_alignment = {}

        for src_idx, tgt_idx in aligned_words:
            if src_idx < len(source_tokens) and tgt_idx < len(target_tokens):
                english_to_hindi_alignment[src_idx] = tgt_idx
                target_ner_tags[tgt_idx] = ner_tags[src_idx]

        # Store final output
        intermediate.append({
            'source_sentence': source_tokens,
            'target_sentence': target_tokens,
            'source_tags': ner_tags,
            'target_tags': target_ner_tags
        })

    # Write JSON
    print(f"[INFO] Writing results to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as write_file:
        json.dump(intermediate, write_file, indent=4, ensure_ascii=False)

    del translator_model
    clear_memory()


if __name__ == "__main__":
    typer.run(main)
