import argparse
import json
import pickle

import pandas as pd
import torch
from tqdm import tqdm

from model import Model
from preprocessing import PADDING_IDX, Preprocessing


def parse_args():
    parser = argparse.ArgumentParser(
        description='Make predictions from input Parquet file.'
    )
    parser.add_argument(
        '--data-path',
        help='Path to the Parquet file containing the test data.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--cfg-path',
        help='Path to the JSON file containing the model configuration.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--model-path',
        help='Path to the model checkpoint.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--preprocessor-path',
        help='Path to the preprocessor.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--output-path',
        help='Path where the predictions will be saved.',
        required=True,
        type=str
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading the preprocessor.")
    with open(args.preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)

    print("Loading the model.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)
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
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)

    print("Loading the test data.")
    test_data = pd.read_parquet(args.data_path, engine='fastparquet')
    if 'index' in test_data.columns:
        test_data.set_index('index', inplace=True)

    print("Making predictions.")
    predictions = pd.Series(index=test_data.index, name='preds', dtype=object)
    model.eval()
    with torch.inference_mode():
        for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0]):
            inputs = preprocessor.transform(row['tokens'])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            predictions.loc[index] = model(inputs).argmax(-1).cpu().tolist()
    predictions.to_frame().to_parquet(args.output_path)
    print(f"The predictions have been saved in {args.output_path}")


if __name__ == '__main__':
    main()
