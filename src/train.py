import argparse
import json
import os
import pickle
import warnings

import pandas as pd
import torch
from tqdm import tqdm

from model import Model
from preprocessing import PADDING_IDX, Preprocessing


class Dataset(torch.utils.data.Dataset):

    def __init__(self, tokens_sequences, labels_sequences, preprocessor):
        super().__init__()
        n_entries = len(tokens_sequences)
        if n_entries != len(labels_sequences):
            raise Exception(
                "The number of tokens sequences does not match the number of labels sequences."
            )
        self.dataset = []
        for tokens, labels in tqdm(zip(tokens_sequences, labels_sequences), total=n_entries, mininterval=1):
            data = preprocessor.transform(tokens)
            data['labels'] = torch.tensor(labels, dtype=torch.long)
            self.dataset.append(data)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument(
        '--data-path',
        help='Path to the Parquet file containing the training data.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--cfg-path',
        help='Path to the JSON file containing the training configuration.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--no-reproducibility',
        help='Use this flag if reproducibility is NOT a concern.',
        default=False,
        type=bool
    )
    parser.add_argument(
        '--model-path',
        help='Path where the model will be saved.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--preprocessor-path',
        help='Path where the preprocessor will be saved.',
        required=True,
        type=str
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Reading the configuration file.")
    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)

    if not args.no_reproducibility:
        print("Making the results reproducible.")
        try:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(0)
        except Exception as e:
            warnings.warn(f"The results might not be reproducible: {e}.")

    print("Reading the input file.")
    df = pd.read_parquet(args.data_path, engine='fastparquet')

    print("Fitting the preprocessor.")
    preprocessor = Preprocessing(
        min_char_freq=cfg['min_char_freq'],
        min_token_freq=cfg['min_token_freq']
    ).fit(df['tokens'])
    with open(args.preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved in {args.preprocessor_path}.")

    print("Generating the dataset.")
    dataset = Dataset(
        df['tokens'], df['token_label_ids'], preprocessor=preprocessor
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True
    )

    print("Initializing the model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(
        n_tokens=preprocessor.n_tokens,
        n_chars=preprocessor.n_chars,
        n_attrs=preprocessor.n_attrs,
        n_classes=cfg['n_classes'],
        padding_idx=PADDING_IDX,
        k_size=cfg['k_size'],
        hidden_dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers'],
        bidirectional=cfg['bidirectional'],
        dropout=cfg['dropout'],
    )
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr_start'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['n_epochs'], eta_min=cfg['lr_end']
    )

    print("Training the model.")
    model.train()
    for epoch in range(1, cfg['n_epochs'] + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg['n_epochs']}")
        running_loss = 0.
        for i, data in enumerate(pbar):
            data = {k: v.to(device) for k, v in data.items()}
            logits = model(data)
            loss = loss_fn(logits, data['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = (i * running_loss +
                            loss.detach().cpu().item()) / (i + 1)
            pbar.set_postfix({'Training Loss': running_loss})
        scheduler.step()
    torch.save(model.state_dict(), args.model_path)
    print(f"The model has been saved in {args.model_path}.")


if __name__ == '__main__':
    main()
