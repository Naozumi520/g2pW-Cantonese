import os
import json
import requests
import zipfile
from io import BytesIO
import shutil

from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import onnxruntime
import numpy as np

from g2pw.dataset import TextDataset, get_phoneme_labels
from g2pw.utils import load_config

def predict(onnx_session, dataloader, labels, turnoff_tqdm=False):
    all_preds = []
    all_confidences = []

    generator = dataloader if turnoff_tqdm else tqdm(dataloader, desc='predict')
    for data in generator:
        input_ids, token_type_ids, attention_mask, phoneme_mask, char_ids, position_ids = \
            [data[name] for name in ('input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids')]

        probs = onnx_session.run(
            [],
            {
                'input_ids': input_ids.numpy(),
                'token_type_ids': token_type_ids.numpy(),
                'attention_mask': attention_mask.numpy(),
                'phoneme_mask': phoneme_mask.numpy(),
                'char_ids': char_ids.numpy(),
                'position_ids': position_ids.numpy()
            }
        )[0]

        preds = np.argmax(probs, axis=-1)
        max_probs = probs[np.arange(probs.shape[0]), preds]

        all_preds += [labels[pred] for pred in preds.tolist()]
        all_confidences += max_probs.tolist()

    return all_preds, all_confidences


class G2PWConverter:
    def __init__(self, model_dir='G2PWModel/', model_source=None, num_workers=None, batch_size=None,
                 turnoff_tqdm=True):

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 2
        self.session_g2pw = onnxruntime.InferenceSession(os.path.join(model_dir, 'g2pw.onnx'), sess_options=sess_options)
        
        self.config = load_config(os.path.join(model_dir, 'config.py'), use_default=True)

        self.num_workers = num_workers if num_workers else self.config.num_workers
        self.batch_size = batch_size if batch_size else self.config.batch_size
        self.model_source = model_source if model_source else self.config.model_source
        self.turnoff_tqdm = turnoff_tqdm

        self.tokenizer = BertTokenizer.from_pretrained(self.model_source)

        polyphonic_chars_path = os.path.join(model_dir, 'POLYPHONIC_CHARS.txt')
        monophonic_chars_path = os.path.join(model_dir, 'MONOPHONIC_CHARS.txt')
        self.polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path).read().strip().split('\n')]
        self.monophonic_chars = [line.split('\t') for line in open(monophonic_chars_path).read().strip().split('\n')]
        self.labels, self.char2phonemes = get_phoneme_labels(self.polyphonic_chars)

        self.chars = sorted(list(self.char2phonemes.keys()))
        self.pos_tags = TextDataset.POS_TAGS

    def __call__(self, sentences):
        if isinstance(sentences, str):
            sentences = [sentences]

        texts, query_ids, sent_ids, partial_results = self._prepare_data(sentences)
        if len(texts) == 0:
            # sentences no polyphonic words
            return partial_results

        dataset = TextDataset(self.tokenizer, self.labels, self.char2phonemes, self.chars, texts, query_ids,
                              use_mask=self.config.use_mask, use_char_phoneme=False,
                              window_size=self.config.window_size, for_train=False)

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.create_mini_batch,
            num_workers=self.num_workers
        )

        preds, confidences = predict(self.session_g2pw, dataloader, self.labels, turnoff_tqdm=self.turnoff_tqdm)

        results = partial_results
        for sent_id, query_id, pred in zip(sent_ids, query_ids, preds):
            results[sent_id][query_id] = pred

        return results

    def _prepare_data(self, sentences):
        polyphonic_chars = set(self.chars)
        monophonic_chars_dict = {
            char: phoneme for char, phoneme in self.monophonic_chars
        }
        texts, query_ids, sent_ids, partial_results = [], [], [], []
        for sent_id, sent in enumerate(sentences):
            partial_result = [None] * len(sent)
            for i, char in enumerate(sent):
                if char in polyphonic_chars:
                    texts.append(sent)
                    query_ids.append(i)
                    sent_ids.append(sent_id)
                elif char in monophonic_chars_dict:
                    partial_result[i] = monophonic_chars_dict[char]
            partial_results.append(partial_result)
        return texts, query_ids, sent_ids, partial_results
