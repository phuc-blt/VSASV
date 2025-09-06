from argparse import ArgumentParser
import time
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from module.dataset import Evaluation_Dataset
from main import Task

def load_checkpoint(ckpt_path, map_location):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        hyper = ckpt.get("hyper_parameters", {})
    else:
        state = ckpt
        hyper = {}
    return state, hyper

def main():
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--embedding_dim', type=int, default=192)
    parser.add_argument('--num_blocks', type=int, default=6)
    parser.add_argument('--loss_name', type=str, default='amsoftmax')
    parser.add_argument('--input_layer', type=str, default='conv2d2')
    parser.add_argument('--pos_enc_layer_type', type=str, default='abs_pos')
    parser.add_argument('--encoder_name', type=str, default=None)
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--trial_path', type=str, required=True)
    parser.add_argument('--second', type=int, default=-1)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--strict_load', action='store_true')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--data_root', type=str, default=None, help='Optional root to prepend to relative wav paths in trial file')
    parser.add_argument('--out_path', type=str, default=None, help='Output file path for <enroll> <test> <cosine_score>')

    hparams = parser.parse_args()
    seed_everything(hparams.seed, workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    state_dict, hyper = load_checkpoint(hparams.checkpoint_path, map_location=device)
    encoder_name = hparams.encoder_name or hyper.get('encoder_name', 'resnet34')
    num_classes = hparams.num_classes or hyper.get('num_classes', 2)

    lightning_model = Task(
        embedding_dim=hparams.embedding_dim,
        num_blocks=hparams.num_blocks,
        loss_name=hparams.loss_name,
        input_layer=hparams.input_layer,
        pos_enc_layer_type=hparams.pos_enc_layer_type,
        trial_path=hparams.trial_path,
        sample_rate=hparams.sample_rate,
        encoder_name=encoder_name,
        num_classes=num_classes,
    )

    load_result = lightning_model.load_state_dict(state_dict, strict=hparams.strict_load)
    if isinstance(load_result, tuple):
        missing, unexpected = load_result
        if not hparams.strict_load:
            if missing:
                print('[load_state_dict] Missing keys (not loaded):', len(missing))
            if unexpected:
                print('[load_state_dict] Unexpected keys (ignored):', len(unexpected))

    lightning_model.to(device)
    lightning_model.eval()
    print(f"Loaded weights from {hparams.checkpoint_path}")

    trials = np.loadtxt(hparams.trial_path, dtype=str)
    if trials.ndim == 1:
        trials = np.expand_dims(trials, 0)
    if trials.shape[1] < 2:
        raise ValueError(f"Trial file {hparams.trial_path} must have at least 2 columns: enroll_path, test_path")

    enroll_paths = trials.T[0]
    test_paths = trials.T[1]

    # Resolve relative paths
    trial_dir = os.path.dirname(hparams.trial_path)

    def resolve_one(p):
        # already absolute
        if os.path.isabs(p) and os.path.exists(p):
            return p
        # try relative to trial file dir
        cand1 = os.path.join(trial_dir, p)
        if os.path.exists(cand1):
            return cand1
        # try data_root
        if hparams.data_root is not None:
            cand2 = os.path.join(hparams.data_root, p)
            if os.path.exists(cand2):
                return cand2
        # return original if none exists; will be filtered later
        return p

    enroll_paths = np.array([resolve_one(p) for p in enroll_paths], dtype=str)
    test_paths = np.array([resolve_one(p) for p in test_paths], dtype=str)

    # Filter out pairs where either path doesn't exist
    exists_mask = np.array([os.path.exists(e) and os.path.exists(t) for e, t in zip(enroll_paths, test_paths)])
    missing_pairs = int((~exists_mask).sum())
    if missing_pairs:
        print(f"[warn] {missing_pairs} pairs reference missing files; they will be skipped")
    enroll_paths = enroll_paths[exists_mask]
    test_paths = test_paths[exists_mask]

    eval_paths = np.unique(np.concatenate((enroll_paths, test_paths)))

    print("number of enroll:", len(set(enroll_paths)))
    print("number of test:", len(set(test_paths)))
    print("number of evaluation:", len(eval_paths))
    print(f"[Eval] Total utterances: {len(eval_paths)}")

    try:
        test_dataset = Evaluation_Dataset(eval_paths, second=hparams.second, sample_rate=hparams.sample_rate)
    except TypeError as e:
        if "unexpected keyword argument 'sample_rate'" in str(e):
            print('[warn] Evaluation_Dataset does not accept sample_rate; falling back without it')
            test_dataset = Evaluation_Dataset(eval_paths, second=hparams.second)
        else:
            raise

    test_loader = DataLoader(
        test_dataset,
        num_workers=hparams.num_workers,
        shuffle=False,
        batch_size=hparams.batch_size,
        pin_memory=(device.type == 'cuda'),
    )

    test_vectors = []
    index_mapping = {}

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            x, path = batch
            p = path[0]
            out = lightning_model(x.to(device))
            vec = out.detach().cpu().numpy()[0]
            test_vectors.append(vec)
            index_mapping[p] = batch_idx

    test_vectors = np.asarray(test_vectors)
    test_vectors = test_vectors - np.mean(test_vectors, axis=0, keepdims=True)

    # Compute cosine similarity scores and save
    lines = []
    for e, t in zip(enroll_paths, test_paths):
        if e not in index_mapping or t not in index_mapping:
            # This should be rare because we prefiltered missing files
            continue
        ve = test_vectors[index_mapping[e]]
        vt = test_vectors[index_mapping[t]]
        cos = np.dot(ve, vt) / (np.linalg.norm(ve) * np.linalg.norm(vt) + 1e-12)
        lines.append(f"{e} {t} {cos:.6f}")

    out_path = hparams.out_path or os.path.join(os.path.dirname(hparams.trial_path), "trials_with_score.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[info] Wrote trials with cosine scores to: {out_path}")

if __name__ == '__main__':
    main()
