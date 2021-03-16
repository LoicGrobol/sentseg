import pathlib

from typing import (
    Any,
    Dict,
    Final,
    IO,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import conllu
import torch

from torch.nn.utils.rnn import pad_sequence

from sentseg import lexers
from sentseg.utils import smart_open

LABELS_LEXICON = {"B": 0, "I": 1, "L": 2}


class TaggedSeq(NamedTuple):
    seq: lexers.BertLexerSentence
    labels: torch.Tensor


_T_SENTDATASET = TypeVar("_T_SENTDATASET", bound="SentDataset")


class SentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sentences: Iterable[Sequence[str]],
        lexer: lexers.BertLexer,
        block_size: int = 128,
        offset: int = 1,
    ):
        self.sentences = [tuple(s) for s in sentences]
        self.lexer = lexer
        self.block_size = block_size
        self.offset = offset

        self._flat_sents: List[str] = []
        self._labels: List[int] = []

        for s in self.sentences:
            self._flat_sents.extend(s)
            if len(s) == 1:
                self._labels.append(LABELS_LEXICON["B"])
            self._labels.extend(
                (
                    LABELS_LEXICON["B"],
                    *(LABELS_LEXICON["I"] for _ in s[1:-1]),
                    LABELS_LEXICON["L"],
                )
            )

    def __len__(self):
        return (len(self._labels) - self.block_size) // self.offset

    def __getitem__(self, index: int):
        seq = self._flat_sents[
            index * self.offset : index * self.offset + self.block_size
        ]
        encoded_seq = self.lexer.encode(seq)
        labels = torch.tensor(
            self._labels[index * self.offset : index * self.offset + self.block_size]
        )
        return TaggedSeq(encoded_seq, labels)

    @classmethod
    def from_conll(
        cls: Type[_T_SENTDATASET],
        filename: Union[str, pathlib.Path, IO[str]],
        config: Optional[Dict[str, Any]] = None,
    ) -> _T_SENTDATASET:
        if config is None:
            config = dict()
        with smart_open(filename) as istream:
            sents = [[t["form"] for t in s] for s in conllu.parse_incr(istream)]

        return cls(sents, **config)


class TaggedSeqBatch(NamedTuple):
    seqs: lexers.BertLexerBatch
    labels: torch.Tensor


class SentLoader(torch.utils.data.DataLoader):
    # Labels that are -100 are ignored in torch crossentropy
    LABEL_PADDING: Final[int] = -100

    def __init__(self, dataset: SentDataset, *args, **kwargs):
        self.dataset: SentDataset
        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = self.collate
        super().__init__(dataset, *args, **kwargs)

    def collate(self, batch: Sequence[TaggedSeq]) -> TaggedSeqBatch:
        seqs_batch = self.dataset.lexer.make_batch([s.seq for s in batch])
        labels_batch = pad_sequence(
            [s.labels for s in batch],
            batch_first=True,
            padding_value=self.LABEL_PADDING,
        )
        return TaggedSeqBatch(seqs_batch, labels_batch)
