import torch
from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
import logging
from IndicTransToolkit import IndicProcessor #type: ignore

# Initialize logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    batch_size: int = 4
    model_checkpoint_dir: str = ''
    source_lang: str = 'eng_Latn'
    target_lang: str = 'hin_Deva'
    q_config: str = '4-bit'
    ip: IndicProcessor =  field(default_factory=IndicProcessor)# type: ignore
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

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

class TranslatorModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        qconfig = init_q_config(self.config.q_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint_dir, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_checkpoint_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=qconfig,
        ).to(self.config.device)

        if qconfig is None:
            self.model.half()

        self.model.eval()

    def translate(self, input_sentence: str, ip):
        input_sentence = ip.preprocess(input_sentence, src_lang=self.config.source_lang, tgt_lang=self.config.target_lang)

        inputs = self.tokenizer(
            input_sentence,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.config.device)

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
            generated_tokens = self.tokenizer.decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

        # Postprocess the translations, including entity replacement
        translations += ip.postprocess(generated_tokens, lang=self.config.target_lang)

        del inputs
        torch.cuda.empty_cache()

        return translations


    def batch_translate(self, input_sentences, ip):
        translations = []
        for i in range(0, len(input_sentences), self.config.batch_size):
            batch = input_sentences[i : i + self.config.batch_size]

            # Preprocess the batch and extract entity mappings
            batch = ip.preprocess_batch(batch, src_lang=self.config.source_lang, tgt_lang=self.config.target_lang)

            # Tokenize the batch and generate input encodings
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.config.device)

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
            translations += ip.postprocess_batch(generated_tokens, lang=self.config.target_lang)

            del inputs
            torch.cuda.empty_cache()

        return translations