# G2pW for Cantonese
A Cantonese Grapheme-to-Phoneme (G2P) conversion tool, adapted from the original G2pW repository for Chinese Mandarin.

## Overview
This is a Cantonese adaptation of the G2pW repository, designed for polyphone disambiguation in Cantonese. The model converts Chinese characters (graphemes) to Jyutping (phonemes).

## Model
The trained model is available at:  
https://huggingface.co/Naozumi0512/G2pW-Cantonese

## Usage
```bash
git clone https://github.com/Naozumi520/G2pW-Cantonese && cd G2pW-Cantonese
git clone https://huggingface.co/hon9kon9ize/bert-large-cantonese
git clone https://huggingface.co/Naozumi0512/g2pW-Cantonese
pip install -r requirements.txt
```
```python
from G2pW import G2pWConverter

conv = G2pWConverter(model_dir='./G2pWModel-v2-onnx/', model_source='./bert-large-cantonese/')
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
This work builds upon the original G2pW model. I would like to express my sincere gratitude to:

- The original `G2pW` authors for their groundbreaking work
- The `rime-cantonese` Project contributors for their extensive Cantonese lexicon
- `Words.hk` team for their comprehensive Cantonese dictionary
- `CantoDict` team for their valuable Cantonese resources
- Our community `hon9kon9ize` for [`hon9kon9ize/bert-large-cantonese`](https://huggingface.co/hon9kon9ize/bert-large-cantonese)

This project is based on the G2pW model:

```bibtex
@inproceedings{chen22d_interspeech,
  title     = {G2pW: A Conditional Weighted Softmax BERT for Polyphone Disambiguation in Mandarin},
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
https://github.com/GitYCC/G2pW/blob/master/LICENCE  
https://github.com/rime/rime-cantonese/blob/main/LICENSE-CC-BY  
https://words.hk/base/hoifong/
