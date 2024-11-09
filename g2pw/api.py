# This code is modified from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/text/g2pw/api.py
# This code is modified from https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw
# This code is modified from https://github.com/GitYCC/g2pW

import warnings
warnings.filterwarnings("ignore")
import json
import os
import zipfile,requests
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import onnxruntime
onnxruntime.set_default_logger_severity(3)
from transformers import BertTokenizer

from .dataset import get_char_phoneme_labels
from .dataset import get_phoneme_labels
from .dataset import prepare_onnx_input
from .utils import load_config

model_version = '1.1'


def predict(session, onnx_input: Dict[str, Any],
            labels: List[str]) -> Tuple[List[str], List[float]]:
    all_preds = []
    all_confidences = []
    probs = session.run([], {
        "input_ids": onnx_input['input_ids'],
        "token_type_ids": onnx_input['token_type_ids'],
        "attention_mask": onnx_input['attention_masks'],
        "phoneme_mask": onnx_input['phoneme_masks'],
        "char_ids": onnx_input['char_ids'],
        "position_ids": onnx_input['position_ids']
    })[0]

    preds = np.argmax(probs, axis=1).tolist()
    max_probs = []
    for index, arr in zip(preds, probs.tolist()):
        max_probs.append(arr[index])
    all_preds += [labels[pred] for pred in preds]
    all_confidences += max_probs

    return all_preds, all_confidences


class G2PWConverter:
    def __init__(self, model_dir='G2PWModel/', model_source=None, num_workers=None, use_cuda=False):

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.intra_op_num_threads = 2
        self.session_g2pW = onnxruntime.InferenceSession(os.path.join(model_dir, 'g2pw.onnx'), sess_options=sess_options, providers=['CUDAExecutionProvider' if use_cuda else 'CPUExecutionProvider'])
        
        self.config = load_config(os.path.join(model_dir, 'config.py'), use_default=True)

        self.model_source = model_source if model_source else self.config.model_source

        self.tokenizer = BertTokenizer.from_pretrained(self.model_source)

        polyphonic_chars_path = os.path.join(model_dir, 'POLYPHONIC_CHARS.txt')
        monophonic_chars_path = os.path.join(model_dir, 'MONOPHONIC_CHARS.txt')
        self.polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path, encoding='utf-8').read().strip().split('\n')]
        self.monophonic_chars = [line.split('\t') for line in open(monophonic_chars_path, encoding='utf-8').read().strip().split('\n')]
        self.labels, self.char2phonemes = get_char_phoneme_labels(
            polyphonic_chars=self.polyphonic_chars
        ) if self.config.use_char_phoneme else get_phoneme_labels(
            polyphonic_chars=self.polyphonic_chars)

        self.chars = sorted(list(self.char2phonemes.keys()))

        self.polyphonic_chars_new = set(self.chars)

        self.monophonic_chars_dict = {
            char: phoneme
            for char, phoneme in self.monophonic_chars
        }

        self.pos_tags = [
            'UNK', 'A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI'
        ]


    def __call__(self, sentences: List[str]) -> List[List[str]]:
        if isinstance(sentences, str):
            sentences = [sentences]

        texts, query_ids, sent_ids, partial_results = self._prepare_data(
            sentences=sentences)
        if len(texts) == 0:
            # sentences no polyphonic words
            return partial_results

        onnx_input = prepare_onnx_input(
            tokenizer=self.tokenizer,
            labels=self.labels,
            char2phonemes=self.char2phonemes,
            chars=self.chars,
            texts=texts,
            query_ids=query_ids,
            use_mask=self.config.use_mask,
            window_size=None)

        preds, confidences = predict(
            session=self.session_g2pW,
            onnx_input=onnx_input,
            labels=self.labels)
        if self.config.use_char_phoneme:
            preds = [pred.split(' ')[1] for pred in preds]

        results = partial_results
        for sent_id, query_id, pred in zip(sent_ids, query_ids, preds):
            results[sent_id][query_id] = pred

        return results

    def _prepare_data(
            self, sentences: List[str]
    ) -> Tuple[List[str], List[int], List[int], List[List[str]]]:
        texts, query_ids, sent_ids, partial_results = [], [], [], []
        for sent_id, sent in enumerate(sentences):
            partial_result = [None] * len(sent)
            for i, char in enumerate(sent):
                if char in self.polyphonic_chars_new:
                    texts.append(sent)
                    query_ids.append(i)
                    sent_ids.append(sent_id)
                elif char in self.monophonic_chars_dict:
                    partial_result[i] = self.monophonic_chars_dict[char]
            partial_results.append(partial_result)
        return texts, query_ids, sent_ids, partial_results