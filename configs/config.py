root = './rimeExtract_dataset/'

manual_seed = 1313
model_source = './bert-large-cantonese'
polyphonic_chars_path = root + 'POLYPHONIC_CHARS.txt'
window_size = 32
num_workers = 2
use_mask = True
use_conditional = True
param_conditional = {
    'bias': True,
    'char-linear': True,
    'pos-linear': False,
    'char+pos-second': True,
}

# for training
exp_name = 'rimeDataLowerLR_BERT_L_DescWS-Sec-cLin-B_POSw01'
train_sent_path = root + 'train.sent'
train_lb_path = root + 'train.lb'
valid_sent_path = root + 'dev.sent'
valid_lb_path = root + 'dev.lb'
test_sent_path = root + 'test.sent'
test_lb_path = root + 'test.lb'
batch_size = 128
# lr = 5e-5
lr = 2e-5
val_interval = 200
num_iter = 30000
use_pos = True
param_pos = {
    'weight': 0.1,
    'pos_joint_training': True,
    'train_pos_path': root + 'train.pos',
    'valid_pos_path': root + 'dev.pos',
    'test_pos_path': root + 'test.pos'
}