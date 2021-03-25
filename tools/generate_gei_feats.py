import argparse
import os
import pickle

import numpy as np
import scipy.io as sio
from src.dataset.dataset import Dataset
from src.utils.image import generate_gei
from tqdm import tqdm

CLASSES = {
    'ADME': 1,
    'AMAN': 2,
    'BEBE': 3,
    'COMP': 4,
    'DADO': 5,
    'DEPR': 6,
    'ELAS': 7,
    'FANT': 8,
    'IMPO': 9,
    'MART': 10,
    'MIGR': 11,
    'NACN': 12,
    'NEVE': 13,
    'NOTC': 14,
    'ORQT': 15,
    'PACN': 16,
    'PARD': 17,
    'REDC': 18,
    'RLAM': 19,
    'SURF': 20,
    'TELE': 21,
    'TRAM': 22,
    'VAPR': 23,
    'VELZ': 24,
}  # it comes from class.mat

# indexes of the 12 words that make part of database12
# it was verified loading a database12....mat
INDEX12 = [1, 6, 8, 10, 12, 13, 16, 17, 18, 19, 22, 23]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate datasets")
    parser.add_argument("--data-dir",
                        type=str,
                        default="/home/wesley.passos/repos/libras/data/database_convertida",
                        help="Videos directory")

    parser.add_argument("--output-dir",
                        type=str,
                        default="/home/wesley.passos/repos/libras/data/feats",
                        help="Directory to save the database generated")

    parser.add_argument("--output-format",
                        type=str,
                        default="python",
                        help="Prepare database to use in python or matlab")

    parser.add_argument("--dim",
                        type=int,
                        default=None,
                        help="Dimension to rescale GEI. If not specified, do not perform rescale.")

    parser.add_argument("--words",
                        type=int,
                        default=24,
                        help="How many words to include (must be 12 or 24)")

    args = parser.parse_args()

    nb_words = int(args.words)

    if args.dim is not None:
        dim = (int(args.dim), int(args.dim))
    else:
        dim = 'same'

    if nb_words != 12 and nb_words != 24:
        raise ValueError("words must be 12 or 24: ", nb_words)

    dataset = Dataset(args.data_dir)

    # TODO: hardcoded. Think a way to put it in argparse
    body_parts = [
        'RightHand',
        'LeftHand',
        'UpperArmLeft',
        'UpperArmRight',
        'LowerArmLeft',
        'LowerArmRight',
        'Head',
    ]

    database = []
    for video in tqdm(dataset, total=len(dataset)):

        # if dataset12, ignore videos with labels indexed outside of INDEX12
        if nb_words == 12 and CLASSES[video.word] not in INDEX12:
            continue

        gei = generate_gei(video, output_dim=dim, body_parts=body_parts)
        label = CLASSES[video.word]

        data = dict()
        data['sample'] = gei
        data['label'] = label

        database.append(data)

    os.makedirs(args.output_dir, exist_ok=True)

    out_filename = f'database{nb_words}_gei_{gei.shape[0]}x{gei.shape[1]}'

    # save in .mat format
    if args.output_format.lower() == 'matlab':
        dt = np.dtype([('sample', np.float32, (gei.shape[0], gei.shape[1])),
                       ('label', (np.double, 1))])
        arr = np.zeros((len(database), ), dtype=dt)
        for idx, d in enumerate(database):
            arr[idx]['sample'] = d['sample']
            arr[idx]['label'] = d['label']

        sio.savemat(os.path.join(args.output_dir, f"{out_filename}.mat"), {'database': arr})

    # save in pkl format
    elif args.output_format.lower() == 'python':
        with open(os.path.join(args.output_dir, f"{out_filename}.pkl"), 'wb') as f:
            pickle.dump(database, f, pickle.HIGHEST_PROTOCOL)

    else:
        raise ValueError(f"Invalid output format {args.output_format}. Choose Matlab or Python")
