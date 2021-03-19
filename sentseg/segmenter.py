from __future__ import annotations

import math
import pathlib
from typing import (
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from boltons import iterutils as itu
from boltons.dictutils import OneToOne
import fast_transformers.builders
import fast_transformers.masking
from loguru import logger
import pydantic
import pytorch_lightning as pl
import pytorch_lightning.metrics as pl_metrics
import toml
import torch
import torch.nn
import transformers

from sentseg import lexers

T = TypeVar("T")

_T_Segmenter = TypeVar("_T_Segmenter", bound="Segmenter")


class Segmenter(torch.nn.Module):
    labels_lexicon = OneToOne({"B": 0, "I": 1, "L": 2})

    def __init__(
        self,
        lexer: lexers.BertLexer,
        depth: int = 1,
        n_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lexer = lexer
        self.depth = depth
        self.dropout = torch.nn.Dropout(dropout)
        self.n_heads = n_heads
        self.transformer = (
            fast_transformers.builders.TransformerEncoderBuilder.from_kwargs(
                n_layers=self.depth,
                n_heads=self.n_heads,
                query_dimensions=self.lexer.out_dim,
                value_dimensions=self.lexer.out_dim,
                feed_forward_dimensions=4 * self.lexer.out_dim,
                attention_type="full",
                activation="gelu",
            ).get()
        )
        self.output_layer = torch.nn.Linear(self.lexer.out_dim, 3)

    def forward(self, inpt: lexers.BertLexerBatch) -> torch.Tensor:
        encoded_inpt = self.lexer(inpt)
        # Pytorch transformers use seq×batch×feats dimensions, I don't understand why
        feats = self.transformer(
            encoded_inpt,
            length_mask=fast_transformers.masking.LengthMask(inpt.sent_lengths),
        )
        feats = self.dropout(feats)
        label_scores = self.output_layer(feats)
        return label_scores

    def segment(
        self,
        words: Iterable[str],
        block_size: int = 128,
        batch_size: int = 1,
        diagnostic: bool = False,
        to_segment: Optional[Iterable[T]] = None,
    ) -> Union[
        Generator[List[T], None, None], Generator[List[Tuple[T, str]], None, None]
    ]:
        if to_segment is None:
            words = list(words)
            to_segment = words
        labels: List[str] = []
        for batch in itu.chunked(itu.chunked(words, block_size), batch_size):

            encoded_batch = self.lexer.make_batch(
                [self.lexer.encode(block) for block in batch]
            )
            with torch.no_grad():
                batch_out_scores = self(encoded_batch)
                batch_labels_idx = batch_out_scores.argmax(dim=-1)
            labels.extend(
                self.labels_lexicon.inv[label]
                for sent_labels in batch_labels_idx.tolist()
                for label in sent_labels
            )

        if diagnostic:
            to_segment = zip(to_segment, labels)

        current_sent: List[T] = []
        for token, label in zip(to_segment, labels):
            if label == "B":
                if current_sent:
                    logger.info(
                        "Inconsistent predicted label: B in unfinished sentence"
                    )
                    yield current_sent
                current_sent = [token]
            elif label == "I":
                if not current_sent:
                    logger.info(
                        "Inconsistent predicted labels: I before sentence beginning"
                    )
                current_sent.append(token)
            elif label == "L":
                if not current_sent:
                    logger.info(
                        "Inconsistent predicted labels: L before sentence beginning"
                    )
                current_sent.append(token)
                yield current_sent
                current_sent = []
            else:
                raise ValueError(
                    f"Unknown label {label!r}, have you been messing with the vocabulary?"
                )
        if current_sent:
            logger.info(
                "Inconsistent predicted labels: unfinished sentence at document end"
            )
            yield current_sent

    def save(self, model_path: pathlib.Path, save_weights: bool = True):
        model_path.mkdir(exist_ok=True, parents=True)
        with open(model_path / "config.toml", "w") as out_stream:
            toml.dump(
                {
                    "depth": self.depth,
                    "dropout": self.dropout.p,
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

    def update(self, preds: torch.Tensor, target: torch.Tensor):  # type: ignore[override]
        assert preds.shape == target.shape
        mask = target.ne(self.ignore_index)
        if mask.any():
            self.correct += preds.eq(target).logical_and(mask).int().sum()
            self.total += mask.sum()

    def compute(self):
        return self.correct.true_divide(self.total)


class TaggedSeq(NamedTuple):
    seq: lexers.BertLexerSentence
    labels: torch.Tensor


class TaggedSeqBatch(NamedTuple):
    seqs: lexers.BertLexerBatch
    labels: torch.Tensor


class SegmenterTrainHparams(pydantic.BaseModel):
    batch_size: int = 64
    betas: Tuple[float, float] = (0.9, 0.98)
    epsilon: float = 1e-8
    learning_rate: float = 1e-4
    lr_decay_steps: Optional[int] = None
    warmup_steps: Union[float, int] = 0
    weight_decay: Optional[float] = None


# TODO: also train on a MLM objective
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
        self.loss = torch.nn.CrossEntropyLoss()
        self.model = model

        self.save_hyperparameters("config")

    def forward(self, inpt: lexers.BertLexerBatch) -> torch.Tensor:  # type: ignore[override]
        return self.model(inpt)

    def training_step(self, batch: TaggedSeqBatch, batch_idx: int):  # type: ignore[override]
        inpt, labels = batch

        outputs = self(inpt)

        loss = self.loss(outputs.view(-1, outputs.shape[-1]), labels.view(-1))

        preds = torch.argmax(outputs, dim=-1)
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

    # TODO: add mean token overlap ratio as a validation metric
    def validation_step(self, batch: TaggedSeqBatch, batch_idx: int):  # type: ignore[override]
        inpt, labels = batch

        outputs = self(inpt)

        loss = self.loss(outputs.view(-1, outputs.shape[-1]), labels.view(-1))

        preds = torch.argmax(outputs, dim=-1)
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

        if self.trainer.max_steps is None:
            # FIXME: this might result in `max_step==float("inf")`, which is currently not explicitely
            # dealt with
            # FIXME: this  assumes that the number of training batches is constant epoch-wise, which might not be true
            max_steps = self.trainer.max_epochs * self.trainer.num_training_batches
        else:
            max_steps = self.trainer.max_steps

        if isinstance(self.config.warmup_steps, float):
            warmup_steps = math.floor(self.config.warmup_steps * max_steps)
        else:
            warmup_steps = self.config.warmup_steps

        if self.config.lr_decay_steps:
            if self.config.lr_decay_steps == -1:
                num_training_steps = max_steps
            else:
                num_training_steps = self.config.lr_decay_steps

            schedule = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        elif self.config.warmup_steps > 0:
            schedule = transformers.get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
            )
            schedulers = [{"scheduler": schedule, "interval": "step"}]
        else:
            schedulers = []

        return [optimizer], schedulers
