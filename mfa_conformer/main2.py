from argparse import ArgumentParser
import torch
import numpy as np
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from module.feature import Mel_Spectrogram
from module.loader import SPK_datamodule
from loss import softmax, amsoftmax
import score as score

class Task(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.trials = np.loadtxt(self.hparams.trial_path, str)
        self.mel_trans = Mel_Spectrogram()

        from module.ecapa_tdnn import ecapa_tdnn
        self.encoder = ecapa_tdnn(embedding_dim=self.hparams.embedding_dim)

        if self.hparams.loss_name == "amsoftmax":
            self.loss_fun = amsoftmax(
                embedding_dim=self.hparams.embedding_dim, num_classes=self.hparams.num_classes)
        else:
            self.loss_fun = softmax(
                embedding_dim=self.hparams.embedding_dim, num_classes=self.hparams.num_classes)

    def forward(self, x):
        feature = self.mel_trans(x)
        embedding = self.encoder(feature)
        return embedding

    def on_test_epoch_start(self):
        self.index_mapping = {}
        self.eval_vectors = []

    def test_step(self, batch, batch_idx):
        x, path = batch
        path = path[0]
        with torch.no_grad():
            x = self.mel_trans(x)
            self.encoder.eval()
            x = self.encoder(x)
        x = x.detach().cpu().numpy()[0]
        self.eval_vectors.append(x)
        self.index_mapping[path] = batch_idx

    def test_epoch_end(self, outputs):
        eval_vectors = np.stack(self.eval_vectors)
        eval_vectors = eval_vectors - np.mean(eval_vectors, axis=0)

        labels, scores = score.cosine_score(self.trials, self.index_mapping, eval_vectors)

        EER, threshold = score.compute_eer(labels, scores)
        print("\nCosine EER: {:.2f}% at threshold {:.4f}".format(EER * 100, threshold))

        minDCF_2, thr_2 = score.compute_minDCF(labels, scores, p_target=0.01)
        minDCF_3, thr_3 = score.compute_minDCF(labels, scores, p_target=0.001)

        print("MinDCF (p=0.01): {:.4f} at threshold {:.4f}".format(minDCF_2, thr_2))
        print("MinDCF (p=0.001): {:.4f} at threshold {:.4f}".format(minDCF_3, thr_3))

        if self.hparams.score_save_path:
            with open(self.hparams.score_save_path, 'w') as f:
                for trial, s in zip(self.trials, scores):
                    f.write(f"{trial[0]} {trial[1]} {trial[2]} {s:.6f}\n")

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--embedding_dim", type=int, default=192)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--loss_name", type=str, default="amsoftmax")
        parser.add_argument("--encoder_name", type=str, default="ecapa_tdnn")
        parser.add_argument("--second", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--train_csv_path", type=str, required=True)
        parser.add_argument("--trial_path", type=str, required=True)
        parser.add_argument("--score_save_path", type=str, default="result_scores.txt")
        parser.add_argument("--checkpoint_path", type=str, required=False)
        return parser

def cli_main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Task.add_model_specific_args(parser)
    args = parser.parse_args()

    seed_everything(42)
    model = Task(**vars(args))

    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=True)
        print("Loaded checkpoint from", args.checkpoint_path)

    dm = SPK_datamodule(
        train_csv_path=args.train_csv_path,
        trial_path=args.trial_path,
        second=args.second,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pairs=False
    )

    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        default_root_dir='.',
        log_every_n_steps=10
    )
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    cli_main()
