import os
import torch
import numpy as np
from argparse import ArgumentParser

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy  # ✅ Sửa ở đây

from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score, Accuracy

from module.feature import Mel_Spectrogram
from module.loader import SPK_datamodule
from loss import softmax, amsoftmax


class Task(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.mel_trans = Mel_Spectrogram()

        from module.resnet import resnet34
        from module.ecapa_tdnn import ecapa_tdnn
        from module.conformer import conformer

        # Model
        if self.hparams.encoder_name == "resnet34":
            self.encoder = resnet34(embedding_dim=self.hparams.embedding_dim)
        elif self.hparams.encoder_name == "ecapa_tdnn":
            self.encoder = ecapa_tdnn(embedding_dim=self.hparams.embedding_dim)
        elif self.hparams.encoder_name == "conformer":
            self.encoder = conformer(
                embedding_dim=self.hparams.embedding_dim,
                num_blocks=self.hparams.num_blocks,
                input_layer=self.hparams.input_layer
            )
        else:
            raise ValueError("Invalid encoder")

        # Loss
        if self.hparams.loss_name == "amsoftmax":
            self.loss_fun = amsoftmax(
                embedding_dim=self.hparams.embedding_dim,
                num_classes=self.hparams.num_classes)
        else:
            self.loss_fun = softmax(
                embedding_dim=self.hparams.embedding_dim,
                num_classes=self.hparams.num_classes)

        # Metrics
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_precision = MulticlassPrecision(num_classes=self.hparams.num_classes, average='macro')
        self.val_recall = MulticlassRecall(num_classes=self.hparams.num_classes, average='macro')
        self.val_f1 = MulticlassF1Score(num_classes=self.hparams.num_classes, average='macro')

    def forward(self, x):
        x = self.mel_trans(x)
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        waveform, label = batch
        feat = self.mel_trans(waveform)
        emb = self.encoder(feat)
        loss, acc = self.loss_fun(emb, label)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveform, label = batch
        feat = self.mel_trans(waveform)
        emb = self.encoder(feat)
        logits = self.loss_fun.forward_classifier(emb)
        preds = torch.argmax(logits, dim=1)
        loss = torch.nn.functional.cross_entropy(logits, label)

        self.val_accuracy.update(preds, label)
        self.val_precision.update(preds, label)
        self.val_recall.update(preds, label)
        self.val_f1.update(preds, label)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return {"loss": loss, "preds": preds, "labels": label}

    def on_validation_epoch_end(self):
        acc = self.val_accuracy.compute()
        pre = self.val_precision.compute()
        rec = self.val_recall.compute()
        f1 = self.val_f1.compute()
        f2 = (5 * pre * rec) / (4 * pre + rec + 1e-8)

        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision", pre)
        self.log("val_recall", rec)
        self.log("val_f1", f1)
        self.log("val_f2", f2)

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--embedding_dim", type=int, default=192)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--num_blocks", type=int, default=6)
        parser.add_argument("--input_layer", type=str, default="conv2d")
        parser.add_argument("--step_size", type=int, default=1)
        parser.add_argument("--gamma", type=float, default=0.9)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--weight_decay", type=float, default=1e-6)
        parser.add_argument("--loss_name", type=str, default="amsoftmax")
        parser.add_argument("--encoder_name", type=str, default="ecapa_tdnn")
        parser.add_argument("--second", type=int, default=3)

        parser.add_argument("--train_csv_path", type=str, required=True)
        parser.add_argument("--save_dir", type=str, required=True)
        parser.add_argument("--checkpoint_path", type=str, default=None)
        parser.add_argument("--aug", action="store_true")
        return parser


def main():
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Task.add_model_specific_args(parser)
    args = parser.parse_args()

    seed_everything(42)
    model = Task(**vars(args))

    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint_path}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Callbacks
    ckpt_callback = ModelCheckpoint(
        monitor="val_loss", mode="min",
        filename="best", save_top_k=1, save_last=True,
        dirpath=args.save_dir
    )
    lr_callback = LearningRateMonitor(logging_interval="step")

    dm = SPK_datamodule(
        train_csv_path=args.train_csv_path,
        trial_path=None,
        second=args.second,
        aug=args.aug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pairs=False
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        gpus=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False),  # ✅ Dùng DDPStrategy
        callbacks=[ckpt_callback, lr_callback],
        default_root_dir=args.save_dir,
        sync_batchnorm=True,
        log_every_n_steps=25,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
