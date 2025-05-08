import torch
import typer
from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
# from IndicTransToolkit.IndicTransToolkit import IndicProcessor #type: ignore

## Local imports
from utils import setup_logging
from utils import load_translator_config

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class TranslatorConfig:
    batch_size: int = 4
    model_checkpoint_dir: str = 'ai4bharat/indictrans2-en-indic-1B'
    source_lang: str = 'eng_Latn'
    target_lang: str = 'hin_Deva'
    file_path: str = "/content/input.txt"
    q_config: str = '4-bit'


lang_to_tags = {
    'en': 'eng_Latn',
    'hi': 'hin_Deva',
    'ur': 'urd_Arab',
    'te': 'tel_Telu',
    'ta': 'tam_Taml'
}

def init_q_config(q_config):
    if q_config == "4-bit":
        qconfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif q_config == "8-bit":
        qconfig = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        qconfig = None
    
    return qconfig

class BaseTranslatorModel:
    def __init__(self, config: TranslatorConfig):
        self.config = config
        qconfig = init_q_config(self.config.q_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint_dir, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_checkpoint_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig
        ).to(device)
        self.ip = None
        # self.ip = IndicProcessor(inference=True) #type:ignore

        if qconfig is None:
            self.model.half()

        self.model.eval()

    def batch_translate(self, input_sentences):
        translations = []
        for i in range(0, len(input_sentences), self.config.batch_size):
            batch = input_sentences[i : i + self.config.batch_size]

            # Preprocess the batch and extract entity mappings
            batch = self.ip.preprocess_batch(batch, src_lang=self.config.source_lang, tgt_lang=self.config.target_lang)

            # Tokenize the batch and generate input encodings
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)

            # Generate translations using the model
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            # Decode the generated tokens into text
            with self.tokenizer.as_target_tokenizer():
                generated_tokens = self.tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            # Postprocess the translations, including entity replacement
            translations += self.ip.postprocess_batch(generated_tokens, lang=self.config.target_lang)

            del inputs
            torch.cuda.empty_cache()

        return translations
    
    def write_to_file(self, results):
        with open(self.config.file_path, 'w', encoding='utf-8') as ofile:
            if isinstance(results, str): 
                pass
            else:
                pass

def main(
    config_name: str = 'config.yaml'
):
    
    translator_config = load_translator_config(config_name)
    source_lang_sentence = 'When I was young, I used to go to the park every day.'

    translator_config = TranslatorConfig(
        batch_size=translator_config.get('batch_size', 4),
        model_checkpoint_dir=translator_config.get('model_checkpoint_dir', ''),
        source_lang=lang_to_tags[translator_config.get('source_lang', '')],
        target_lang=lang_to_tags[translator_config.get('target_lang', '')],
        q_config=translator_config.get('q_config', ''),
        device=translator_config.get('device', 'cpu')
    )

    print("Loading logger config")
    setup_logging('translator_model.log')

    choice = typer.prompt(
        "Select an option:\n[1] Translate File\n[2] Align Sentences\n[3] Exit",
        type=int
    )

    if choice == 1:
        typer.echo("Running Translator...")
    elif choice == 2:
        typer.echo("🧬 Running Aligner...")
    elif choice == 3:
        typer.echo("👋 Exiting. Bye!")
        raise typer.Exit()
    else:
        typer.secho("❌ Invalid choice", fg=typer.colors.RED)

    # # Model init
    # translator_model = BaseTranslatorModel(config=translator_config)

    # # Translate a sentence
    # translated = translator_model.translate(source_lang_sentence)
    # print(f"Translation: {translated}")

if __name__ == "__main__":
    typer.run(main)