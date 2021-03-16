import pathlib
from typing import Optional, Tuple, Type, TypeVar

import pydantic
import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_metrics
import toml
import torch
import torch.nn
import transformers

from sentseg import lexers

_T_Segmenter = TypeVar("_T_Segmenter", bound="Segmenter")


class Segmenter(torch.nn.Module):
    def __init__(self, lexer: lexers.BertLexer, depth: int = 1, n_heads: int = 1):
        super().__init__()
        self.lexer = lexer
        self.transformer = torch.nn.Transformer(
            d_model=self.lexer.out_dim,
            dim_feedforward=4 * self.lexer.out_dim,
            nhead=n_heads,
            num_encoder_layers=depth,
            num_decoder_layers=0,
        )
        self.output_layer = torch.nn.Linear(self.lexer.out, 3)

    def forward(self, inpt: lexers.BertLexerBatch) -> torch.Tensor:
        encoded_inpt = self.lexer(inpt)
        feats = self.transformer(encoded_inpt, src_key_padding_mask=inpt.padding_mask)
        label_scores = self.output_layer(feats)
        return label_scores

    def save(self, model_path: pathlib.Path, save_weights: bool = True):
        model_path.mkdir(exist_ok=True, parents=True)
        with open(model_path / "config.toml", "w") as out_stream:
            toml.dump(
                {
                    "depth": self.depth,
                    "n_heads": self.n_heads,
                },
                out_stream,
            )
        self.lexer.save(model_path / "lexer", save_weights=False)
        if save_weights:
            torch.save(self.state_dict(), model_path / "weights.pt")

    @classmethod
    def load(cls: Type[_T_Segmenter], model_path: pathlib.Path) -> _T_Segmenter:
        with open(model_path / "config.toml") as in_stream:
            config = toml.load(in_stream)
        lexer = lexers.BertLexer.load(model_path / "lexer")
        res = cls(lexer=lexer, **config)
        weights_path = model_path / "weights.pt"
        if weights_path.exists():
            res.load_state_dict(torch.load(model_path / "weights.pt"))
        return res


class MaskedAccuracy(pl_metrics.Metric):
    def __init__(self, ignore_index: int = -100):
        super().__init__()

        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        mask = target.ne(self.ignore_index)
        if mask.any():
            self.correct += preds.eq(target).logical_and(mask).int().sum()
            self.total += mask.sum()

    def compute(self):
        return self.correct.true_divide(self.total)


class SegmenterTrainHparams(pydantic.BaseModel):
    batch_size: int = 64
    betas: Tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-8
    learning_rate: float = 1e-4
    lr_decay_steps: Optional[int] = None
    warmup_steps: int = 0
    weight_decay: Optional[float] = None


class SegmenterTrainModule(pl.LightningModule):
    def __init__(
        self,
        model: Segmenter,
        config: Optional[SegmenterTrainHparams] = None,
    ):
        super().__init__()

        if config is not None:
            self.config = config
        else:
            self.config = SegmenterTrainHparams()

        self.accuracy = MaskedAccuracy()
        self.model = model

        self.save_hyperparameters("config")

    def forward(self, inpt: lexers.BertLexerBatch) -> torch.Tensor:
        return self.model(inpt)

    def training_step(self, batch: lexers.TaggedSeqBatch, batch_idx: int):
        inpt, labels = batch

        outputs = self(inpt)

        loss = outputs.loss

        preds = torch.argmax(outputs.logits, dim=-1)
        accuracy = self.accuracy(preds, labels)

        self.log(
            "train/loss",
            loss,
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/accuracy",
            accuracy,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: lexers.TaggedSeqBatch, batch_idx: int):
        inpt, labels = batch

        outputs = self(inpt)

        loss = outputs.loss

        preds = torch.argmax(outputs.logits, dim=-1)
        accuracy = self.accuracy(preds, labels)

        self.log(
            "validation/loss",
            loss,
            reduce_fx=torch.mean,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "validation/accuracy",
            accuracy,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self):
        if self.config.weight_decay is not None:
            no_decay = ["bias", "LayerNorm.weight"]
            decay_rate = self.config.weight_decay
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": decay_rate,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            decay_rate = 0.0
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters()],
                    "weight_decay": 0.0,
                },
            ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=self.config.betas,
            lr=self.config.learning_rate,
            eps=self.config.epsilon,
            weight_decay=decay_rate,
        )
        if self.config.lr_decay_steps:
            if self.config.lr_decay_steps == -1:
                num_training_steps = self.trainer.max_steps
            else:
                num_training_steps = self.config.lr_decay_steps

            schedule = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        elif self.config.warmup_steps > 0:
            schedule = transformers.get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        else:
            schedulers = []

        return [optimizer], schedulers
