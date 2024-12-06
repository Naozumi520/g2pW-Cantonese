# g2pW for Cantonese
A Cantonese Grapheme-to-Phoneme (G2P) conversion tool, adapted from the original g2pW repository for Chinese Mandarin.

## Overview
This is a Cantonese adaptation of the g2pW repository, designed for polyphone disambiguation in Cantonese. The model converts Chinese characters (graphemes) to Jyutping (phonemes).

## Model
The trained model is available at:  
https://huggingface.co/Naozumi0512/g2pW-canto-20241206-bert-base

## Usage
```bash
git clone https://github.com/Naozumi520/g2pW-Cantonese && cd g2pW-Cantonese
git switch 20241206-bert-base
git clone https://huggingface.co/hon9kon9ize/bert-base-cantonese
git clone https://huggingface.co/Naozumi0512/g2pW-canto-20241206-bert-base G2PWModel-v2-onnx
pip install -r requirements.txt
```
```python
from g2pw import G2PWConverter

conv = G2PWConverter(model_dir='./G2PWModel-v2-onnx/', model_source='./bert-base-cantonese/', use_cuda=True)
# "校":
sentence = '調校溫度'
print(conv(sentence)) # [['tiu4', 'gaau3', 'wan1', 'dou6']]
sentence_1 = '校服'
print(conv(sentence_1)) # [['haau6', 'fuk6']]
```

## Data Sources
This model was trained on data collected from:
- [Rime Cantonese Input Schema](https://github.com/rime/rime-cantonese) (jyut6ping3.dict.yaml)
- [粵典 Words.hk](https://words.hk/)
- [CantoDict](https://cantonese.sheik.co.uk/)


## Acknowledgements
This work builds upon the original g2pW model. I would like to express my sincere gratitude to:

- The original `g2pW` authors for their groundbreaking work
- The `rime-cantonese` Project contributors for their extensive Cantonese lexicon
- `Words.hk` team for their comprehensive Cantonese dictionary
- `CantoDict` team for their valuable Cantonese resources
- Our community `hon9kon9ize` for [`hon9kon9ize/bert-base-cantonese`](https://huggingface.co/hon9kon9ize/bert-base-cantonese)

This project is based on the g2pW model:

```bibtex
@inproceedings{chen22d_interspeech,
  title     = {g2pW: A Conditional Weighted Softmax BERT for Polyphone Disambiguation in Mandarin},
  author    = {Yi-Chang Chen and Yu-Chuan Steven and Yen-Cheng Chang and Yi-Ren Yeh},
  year      = {2022},
  booktitle = {Interspeech 2022},
  pages     = {1926--1930},
  doi       = {10.21437/Interspeech.2022-216},
  issn      = {2958-1796},
}
```

For the Cantonese adaptation specifically, please also acknowledge the data sources mentioned above.

## License
https://github.com/GitYCC/g2pW/blob/master/LICENCE  
https://github.com/rime/rime-cantonese/blob/main/LICENSE-CC-BY  
https://words.hk/base/hoifong/
