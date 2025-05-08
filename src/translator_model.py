import torch
from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from IndicTransToolkit.IndicTransToolkit import IndicProcessor #type: ignore

from utils import get_device

@dataclass
class TranslatorConfig:
    batch_size: int = 4
    model_checkpoint_dir: str = 'ai4bharat/indictrans2-en-indic-1B'
    source_lang: str = 'eng_Latn'
    target_lang: str = 'hin_Deva'
    q_config: str = '4-bit'

device = get_device()

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

        self.ip = IndicProcessor(inference=True)

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