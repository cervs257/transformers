import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter 

from tqdm import tqdm

from dataset import BilingualDataset, causal_mask
from transformers import Transformer, build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizer.models import WordLevel
from tokenizer.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace



from pathlib import Path

def get_all_sentences(dataset, lang):
    for example in dataset:
        yield example['translation'][lang]

def get_or_build_tokenizer(config, dataset, lang):
    # config['tokenizer_file'] = '../tokenizers/tokenizer?{0}.json
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    dataset_raw = load_dataset('opus_books', f'{config['lang_src']}-{config['lang_tgt']}', split = 'train')

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    # keep 90% for training and 10% for validation
    train_dataset_size = int(0.9*len(dataset_raw))
    val_dataset_size = len(dataset_raw) - train_dataset_size
    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_dataset_size, val_dataset_size])

    train_dataset = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_dataset = BilingualDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source text: {max_len_src}")
    print(f"Max length of target text: {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = False) # batch size 1 so that i process each sentence one by one

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len = vocab_src_len,
        vocab_tgt_len = vocab_tgt_len,
        src_seq_len = config['seq_len'],
        tgt_seq_len = config['seq_len'],
        d_model = config['d_model'],
        # rest are defaults
    )
    return model


def train_model(config):
    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # make sure weights folder exists
    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    # restore state of model & optimizer if crashes
    initial_epoch = 0
    global_step = 0

    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1).to(device) # smoothing means from every high prob token, take 0.1 prob and distribute it to all other tokens

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Epoch {epoch}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, 1, 1, seq_len) .. hide only padding tokens
            decoder_mask = batch['decoder_mask'].to(device) # (batch_size, 1, seq_len, seq_len) .. also hide subsequent tokens

            # run tensors through the Transformer
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (batch_size, seq_len, d_model)
            proj_output = model.project(decoder_output) # (batch_size, seq_len, vocab_tgt_len)

            # compare output with label
            label = batch['label'].to(device) # (batch_size, seq_len)
            # (batch_size, seq_len, vocab_tgt_len) -> (batch_size*seq_len, vocab_tgt_len
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # backpropagation
            loss.backward()

            # update weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # save model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

if __name__ == '__main__':
    # warning.filterwarnings('ignore')
    config = get_config()
    train_model(config)

