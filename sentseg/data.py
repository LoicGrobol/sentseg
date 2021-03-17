from __future__ import annotations

import pathlib

from typing import (
    Dict,
    Final,
    IO,
    Iterable,
    List,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import conllu
import torch

from torch.nn.utils.rnn import pad_sequence

from sentseg import segmenter as segmod
from sentseg.utils import smart_open


def vocab_from_conllu(filename: Union[str, pathlib.Path, IO[str]]) -> Dict[str, int]:
    with smart_open(filename) as istream:
        res = {
            tok: idx
            for idx, tok in enumerate(
                sorted(set(t["form"] for s in conllu.parse_incr(istream) for t in s))
            )
        }
    return res


_T_SENTDATASET = TypeVar("_T_SENTDATASET", bound="SentDataset")


class SentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sentences: Iterable[Sequence[str]],
        segmenter: segmod.Segmenter,
        block_size: int = 128,
        offset: int = 1,
    ):
        self.sentences = [tuple(s) for s in sentences]
        self.segmenter = segmenter
        self.block_size = block_size
        self.offset = offset

        self._flat_sents: List[str] = []
        self._labels: List[int] = []

        for s in self.sentences:
            self._flat_sents.extend(s)
            if len(s) == 1:
                self._labels.append(self.segmenter.labels_lexicon["B"])
            self._labels.extend(
                (
                    self.segmenter.labels_lexicon["B"],
                    *(self.segmenter.labels_lexicon["I"] for _ in s[1:-1]),
                    self.segmenter.labels_lexicon["L"],
                )
            )

    def __len__(self):
        return (len(self._labels) - self.block_size) // self.offset

    def __getitem__(self, index: int):
        seq = self._flat_sents[
            index * self.offset : index * self.offset + self.block_size
        ]
        encoded_seq = self.segmenter.lexer.encode(seq)
        labels = torch.tensor(
            self._labels[index * self.offset : index * self.offset + self.block_size]
        )
        return segmod.TaggedSeq(encoded_seq, labels)

    @classmethod
    def from_conllu(
        cls: Type[_T_SENTDATASET], filename: Union[str, pathlib.Path, IO[str]], **kwargs
    ) -> _T_SENTDATASET:
        with smart_open(filename) as istream:
            sents = [[t["form"] for t in s] for s in conllu.parse_incr(istream)]

        return cls(sents, **kwargs)


class SentLoader(torch.utils.data.DataLoader):
    # Labels that are -100 are ignored in torch crossentropy
    LABEL_PADDING: Final[int] = -100

    def __init__(self, dataset: SentDataset, *args, **kwargs):
        self.dataset: SentDataset
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = self.collate
        super().__init__(dataset, *args, **kwargs)

    def collate(self, batch: Sequence[segmod.TaggedSeq]) -> segmod.TaggedSeqBatch:
        seqs_batch = self.dataset.segmenter.lexer.make_batch([s.seq for s in batch])
        labels_batch = pad_sequence(
            [s.labels for s in batch],
            batch_first=True,
            padding_value=self.LABEL_PADDING,
        )
        return segmod.TaggedSeqBatch(seqs_batch, labels_batch)
