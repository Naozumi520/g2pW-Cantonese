import pycantonese
from tqdm import tqdm
import numpy as np

root = '../rimeExtract_dataset/'
slices = 'dev'
path = f'{root}/{slices}' + '.sent'
out_path = f'{root}/{slices}' + '_hkcancor.pos'

ANCHOR_CHAR = '▁'
batch_size = 2048

mapping = {
    'ADJ': 'ADJ',
    'ADP': 'ADP',
    'ADV': 'ADV',
    'AUX': 'AUX',
    'CCONJ': 'C',
    'DET': 'D',
    'INTJ': 'I',
    'NOUN': 'N',
    'NUM': 'NUM',
    'PART': 'PT',
    'PRON': 'PN',
    'PROPN': 'PPN',
    'PUNCT': 'PU',
    'SCONJ': 'SC',
    'SYM': 'SYM',
    'VERB': 'V',
    'X': 'X'
}

lines = open(path).read().strip().split('\n')
sentence_list = [line.replace(ANCHOR_CHAR, '') for line in lines]
position_list = [line.index(ANCHOR_CHAR) for line in lines]

fw = open(out_path, 'w', buffering=batch_size)

i = 0

for _ in tqdm(list(range(int(np.ceil(len(lines) / batch_size))))):
    j = min(len(lines), i + batch_size)
    sentences = [sent for sent in sentence_list[i:j]]
    positions = position_list[i:j]

    annotations = []
    for position, sentence in zip(positions, sentences):
        words = list(sentence)
        pos_tags = pycantonese.pos_tag(words)

        annotation = None
        x = -1
        for word, tag in pos_tags:
            for _ in range(len(word)):
                x += 1
                if x == position:
                    annotation = mapping.get(tag, 'UNK')
                    break
            if annotation:
                break
        assert annotation is not None
        annotations.append(annotation)

    if i > 0:
        fw.write('\n')
    fw.write('\n'.join(annotations))
    i = j

fw.close()
