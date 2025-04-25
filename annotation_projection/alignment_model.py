from dataclasses import dataclass
import transformers
import torch
import itertools
import logging
from sentence_transformers import SentenceTransformer

def init_logger():
    pass

init_logger()

@dataclass
class Config:
    model: str = 'bert-base-multilingual-cased'
    align_layer: int = 8
    source_lang: str = 'en'
    target_lang: str = 'hi'
    threshold = 1e-3
    is_st: bool = False
    probs_layer: str = 'softmax'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class BaseAlignmentModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = transformers.BertModel.from_pretrained(self.config.model).to(self.config.device)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.config.model)

        ## Set it to eval mode for inference
        self.model.eval()

    def _preprocess_target(self, tgt):
        if isinstance(tgt, str):
            sent_tgt = tgt.strip().split()
        elif isinstance(tgt, list):
            sent_tgt = [t.strip().split() for t in tgt]

        return sent_tgt

    def _preprocess_source(self, src):
        if isinstance(src, str):
            sent_src = src.strip().split()
        elif isinstance(src, list):
            sent_src = [s.strip().split() for s in src]
            
        return sent_src

    def get_alignments(self, source, target):
        tokens_src = [self.tokenizer.tokenize(word) for word in source]
        tokens_tgt = [self.tokenizer.tokenize(word) for word in target]

        word_ids_src = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens_src]
        word_ids_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens_tgt]

        ids_src = self.tokenizer.prepare_for_model(
            list(itertools.chain(*word_ids_src)), return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length
        )['input_ids']

        ids_tgt = self.tokenizer.prepare_for_model(
            list(itertools.chain(*word_ids_tgt)), return_tensors='pt', truncation=True, max_length=self.tokenizer.model_max_length
        )['input_ids']

        sub2word_map_src = [i for i, word_list in enumerate(tokens_src) for _ in word_list]
        sub2word_map_tgt = [i for i, word_list in enumerate(tokens_tgt) for _ in word_list]

        with torch.no_grad():
            output_source = self.model(ids_src.to(self.config.device), output_hidden_states=True)[2][self.config.align_layer][0, 1:-1]
            output_target = self.model(ids_tgt.to(self.config.device), output_hidden_states=True)[2][self.config.align_layer][0, 1:-1]

            dot = torch.matmul(output_source, output_target.transpose(-1, -2))

            sm_src_tgt = torch.nn.functional.softmax(dot, dim=-1)
            sm_tgt_src = torch.nn.functional.softmax(dot, dim=-2)

            softmax_inter = (sm_src_tgt > self.config.threshold) * (sm_tgt_src > self.config.threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
        align_words = {
            (sub2word_map_src[i.item()], sub2word_map_tgt[j.item()])
            for i, j in align_subwords
        }

        return align_words

    def write_to_file(self):
        pass        

    def get_batch_alignments(self):
        pass