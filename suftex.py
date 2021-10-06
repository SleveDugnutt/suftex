import os
import sys
import time
import argparse
import pandas as pd
import sentencepiece as spm
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from performer_pytorch import PerformerLM, AutoregressiveWrapper

def tokenizer(text, sp):
    return sp.EncodeAsPieces(text)

def calc_max_len(src, sp):
    dst = [tokenizer(s, sp) for s in src]
    return max(len(d) for d in dst)

def add_eos_and_pad(data, max_seq_len, eos, pad, sp):
    output = []
    for d in tqdm(data):
        d = tokenizer(d, sp)
        #add eos token
        d.append(eos)
        #pad sequences
        for i in range(len(d), max_seq_len):
            d.append(pad)
        d = ''.join(d)
        output.append(encode_tokens(d))
    return output

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

def to_tensor(data):
    output = []
    for seq in tqdm(data):
        seq_tensor = torch.tensor(seq).long()
        output.append(seq_tensor)
    return torch.stack(output)

def make_dataloader(data, batch_size, shuffle=True):
    data_tensor = to_tensor(data)
    dataset = MyDataset(data_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def build_model(parameter_dict, device):
    num_tokens = parameter_dict['num_tokens']
    max_seq_len = parameter_dict['max_seq_len']
    dim = parameter_dict['dim']
    depth = parameter_dict['depth']
    heads = parameter_dict['heads']
    lr = parameter_dict['lr']
    model = PerformerLM(
                num_tokens = num_tokens,
                max_seq_len = max_seq_len,
                dim = dim,
                depth = depth,
                heads = heads,
                causal = True,
                reversible = True,
                use_scalenorm = True
            ).to(device)
    model = AutoregressiveWrapper(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

def save_model(path, model, optimizer, train_loss, val_loss, parameter_dict):
    torch.save({
        "model_state_dict" : model.state_dict(),
        "optim_state_dict" : optimizer.state_dict(),
        "train_loss" : train_loss,
        "val_loss" : val_loss,
        "parameter_dict" : parameter_dict
    }, path)

def eval_model(model, data_loader, batch_size):
    model.eval()
    val = 0
    num_batches = len(data_loader.dataset) // batch_size + 1
    with torch.no_grad():
        for _, data in enumerate(tqdm(data_loader)):
            data = data.to(device)
            loss = model(data, return_loss=True)
            val += loss.item()
        val /= num_batches
    return val

def train_model(model, train_data, test_data, batch_size, num_epochs, 
                output_dir, train_loss, val_loss, parameter_dict):
    num_batches = len(train_data.dataset) // batch_size + 1
    start_epoch = len(train_loss)
    start = time.time()
    print('now training')
    for epoch in range(num_epochs):
        model.train()
        print(f'EPOCH {start_epoch + epoch + 1}/{start_epoch + num_epochs}')
        epoch_loss = 0
        for _, data in enumerate(tqdm(train_data)):
            data = data.to(device)
            loss = model(data, return_loss=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= num_batches
        train_loss.append(epoch_loss)
        print('validating')
        val = eval_model(model, test_data, batch_size)
        val_loss.append(val)
        end = time.time() - start
        print(f'TRAIN LOSS {epoch_loss} : VAL LOSS {val} : TIME {end}')
        file_name = f'checkpoint_epoch{start_epoch + epoch + 1}.pt'
        save_model(os.path.join(output_dir, file_name), 
                   model, optimizer, train_loss, val_loss, parameter_dict)
    print('done')

def encode_tokens(text):
    return sp.Encode(text)

def decode_tokens(tokens):
    return sp.Decode(tokens)

def generate_text(input, model, eos_token, seq_len=140, temperature=1.0):
    reversed_input = ''.join(list(reversed(str(input))))
    input_tokens = encode_tokens(reversed_input)
    start_tokens = torch.tensor(input_tokens).long().to(device)
    generated_tokens = model.generate(start_tokens, seq_len, eos_token, temperature).tolist()
    reversed_tokens = [t for t in generated_tokens if t != eos_token]
    reversed_output = decode_tokens(reversed_tokens)
    output = ''.join(list(reversed(reversed_output)))
    return output + input

def get_arguments():
    parser = argparse.ArgumentParser(description='text generation model')
    parser.add_argument('--data', '-d', type=str,
                        help='path to CSV file on which to train a model')
    parser.add_argument('--column', '-col', type=str,
                        help='column to use of csv file')
    parser.add_argument('--sentencepiece', '-sp', type=str,
                        help='path to sentencepiece model file with which to make an encoder and decoder')
    parser.add_argument('--checkpoint', '-cpt', type=str,
                        help='checkpoint to use')
    parser.add_argument('--train_sp', action='store_true',
                        help='train sentencepiece on data file')
    parser.add_argument('--train', action='store_true',
                        help='do train model')
    parser.add_argument('--vocab_size', type=str, default='10000',
                        help='vocab size of sentencepiece : default 10000')
    parser.add_argument('--character_coverage', type=str, default='0.995',
                        help='character coverage of sentencepiece : default 0.995')
    parser.add_argument('--dim', type=int, default=512,
                        help='dimension of a model : default 512')
    parser.add_argument('--depth', type=int, default=4,
                        help='depth of a model : default 4')
    parser.add_argument('--heads', type=int, default=8,
                        help='number of attention heads : default 8')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate : default 1e-4')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='batch size of data : default 32')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='test size of data : default 0.2')
    parser.add_argument('--num_epochs', '-n', type=int, default=10,
                        help='number of train epochs : default 10')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='output directory')
    parser.add_argument('--generate', action='store_true',
                        help='generate texts')
    parser.add_argument('--suffix', '-suf', type=str,
                        help='generate texts ending with this suffix')
    parser.add_argument('--temperatue', type=float, default=1.0,
                         help='temperature when generating texts : default 1.0')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    path = args.data
    output_dir = args.output_dir
    column = args.column
    #train sentencepiece
    if args.train_sp:
        texts = pd.read_csv(path)[column].to_list()
        data = []
        for t in texts:
            reversed_text = ''.join(list(reversed(str(t))))
            data.append(reversed_text)
        df = pd.DataFrame(data, columns=['text'], dtype=str)
        rev_data = 'suftex_train_file_for_sentencepiece.csv'
        df.to_csv(rev_data, index=False)
        vocab_size = args.vocab_size
        coverage = args.character_coverage
        spm.SentencePieceTrainer.train(f'--input={rev_data} \
                                         --model_prefix=suftex \
                                         --vocab_size={vocab_size} \
                                         --character_coverage={coverage} \
                                         --user_defined_symbols=<pad>,<eos>')
        sys.exit()
    
    model_file = args.sentencepiece
    sp = spm.SentencePieceProcessor(model_file)
    pad = '<pad>'
    eos = '<eos>'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.train:
        texts = pd.read_csv(path)[column].to_list()
        data = []
        for t in texts:
            reversed_text = ''.join(list(reversed(str(t))))
            data.append(reversed_text)
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            parameter_dict = checkpoint['parameter_dict']
            max_seq_len = parameter_dict['max_seq_len']
        else:
            max_seq_len = calc_max_len(data, sp) + 1
            num_tokens = len(sp)
            dim = args.dim
            depth = args.depth
            heads = args.heads
            lr = args.lr
            parameter_dict = {
                'num_tokens' : num_tokens,
                'max_seq_len' : max_seq_len,
                'dim' : dim,
                'depth' : depth,
                'heads' : heads,
                'lr' : lr
            }
        data = add_eos_and_pad(data, max_seq_len, eos, pad, sp)
        model, optimizer = build_model(parameter_dict, device)
        test_size = args.test_size
        text_train, text_test = train_test_split(data,
                                                 test_size=test_size,
                                                 random_state=42)
        batch_size = args.batch_size
        train_data = make_dataloader(text_train, batch_size, True)
        test_data = make_dataloader(text_test, batch_size, False)
        num_epochs = args.num_epochs
        output_dir = 'checkpoint'
        if args.output_dir:
            output_dir = os.path.join(args.output_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            train_loss = checkpoint['train_loss']
            val_loss = checkpoint['val_loss']
        else:
            train_loss = []
            val_loss = []
        train_model(model, train_data, test_data, batch_size, num_epochs,
                    output_dir, train_loss, val_loss, parameter_dict)
    if args.generate:
        if not args.suffix:
            print('give suffix')
            sys.exit()
        if not args.checkpoint:
            print('give checkpoint')
            sys.exit()
        checkpoint = torch.load(args.checkpoint)
        parameter_dict = checkpoint['parameter_dict']
        model, _ = build_model(parameter_dict, device)
        input = args.suffix
        eos_token = sp.Encode(eos)[-1]
        temperature = args.temperatue
        max_seq_len = parameter_dict['max_seq_len']
        output = generate_text(input, model, eos_token, max_seq_len, temperature)
        print(output)
