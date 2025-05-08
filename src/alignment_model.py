from dataclasses import dataclass
import transformers
import torch
import itertools

from utils import get_device

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class AlignerConfig:
    model: str = 'bert-base-multilingual-cased'
    output_file: str = 'file.txt'
    align_layer: int = 8
    threshold: float = 1e-3
    probs_layer: str = 'softmax'
    debug_mode: bool = True

device = get_device()

class BaseAlignmentModel:
    def __init__(self, config: AlignerConfig):
        self.config = config
        self.model = transformers.BertModel.from_pretrained(
            self.config.model
        ).to(device)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            self.config.model
        )

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
    
    def _tokenize(self, source, target):
        if isinstance(source, str) and isinstance(target, str):
            sub2word_map_src, sub2word_map_tgt = [], []
            source = self._preprocess_source(source)
            target = self._preprocess_target(target)

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

            for i, word_list in enumerate(tokens_src):
                sub2word_map_src += [i for x in word_list]

            for i, word_list in enumerate(tokens_tgt):
                sub2word_map_tgt += [i for x in word_list]

        else:
            raise TypeError("Check the types please")
        
        return tokens_src, tokens_tgt, word_ids_src, word_ids_tgt, ids_src, ids_tgt, sub2word_map_src, sub2word_map_tgt


    def get_alignments(self, source, target):
        tokens_src, tokens_tgt, word_ids_src, word_ids_tgt, ids_src, ids_tgt, sub2word_map_src, sub2word_map_tgt = self._tokenize(source, target)
        
        with torch.no_grad():
            output_source = self.model(ids_src.to(device).unsqueeze(0), output_hidden_states=True)[2][self.config.align_layer][0, 1:-1]
            output_target = self.model(ids_tgt.to(device).unsqueeze(0), output_hidden_states=True)[2][self.config.align_layer][0, 1:-1]

            dot = torch.matmul(output_source, output_target.transpose(-1, -2))

            sm_src_tgt = torch.nn.functional.softmax(dot, dim=-1)
            sm_tgt_src = torch.nn.functional.softmax(dot, dim=-2)

            softmax_inter = (sm_src_tgt > self.config.threshold) * (sm_tgt_src > self.config.threshold)

        align_subwords = torch.nonzero(softmax_inter, as_tuple=False)

        self.align_words = {
            (sub2word_map_src[i], sub2word_map_tgt[j])
            for i, j in align_subwords
        }

        for i, j in self.align_words:
            print(source.split(" ")[i], target.split(" ")[j])

        return self.align_words

    def write_to_file(self):
        if self.config.debug_mode == False:
            return 

        with open(self.config.output_file, 'w') as file_writer:
            for left_word, right_word in self.align_words:
                file_writer.write(f'[SRC]{left_word} === [TGT]{right_word}')
        