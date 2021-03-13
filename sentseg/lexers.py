import pathlib
from typing import (
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.jit
import torch.nn
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers.tokenization_utils_base import BatchEncoding, TokenSpan
import yaml


_T_BertLexerBatch = TypeVar("_T_BertLexerBatch", bound="BertLexerBatch")


@torch.jit.script
def integer_dropout(t: torch.Tensor, fill_value: int, p: float) -> torch.Tensor:
    mask = torch.empty_like(t, dtype=torch.bool).bernoulli_(p)
    return t.masked_fill(mask, fill_value)


class BertLexerBatch(NamedTuple):
    word_indices: torch.Tensor
    padding_mask: torch.Tensor
    bert_encoding: BatchEncoding
    subword_alignments: Sequence[Sequence[TokenSpan]]

    def to(
        self: _T_BertLexerBatch, device: Union[str, torch.device]
    ) -> _T_BertLexerBatch:
        return type(self)(
            bert_encoding=self.bert_encoding.to(device=device),
            padding_mask=self.padding_mask,
            subword_alignments=self.subword_alignments,
            word_indices=self.word_indices.to(device=device),
        )

    def size(self, *args, **kwargs):
        return self.word_indices.size(*args, **kwargs)


class BertLexerSentence(NamedTuple):
    word_indices: torch.Tensor
    bert_encoding: BatchEncoding
    subwords_alignments: Sequence[TokenSpan]


def align_with_special_tokens(
    word_lengths: Sequence[int],
    mask=Sequence[int],
    special_tokens_code: int = 1,
    sequence_tokens_code: int = 0,
) -> List[TokenSpan]:
    """Provide a word→subwords alignements using an encoded sentence special tokens mask.

    This is only useful for the non-fast 🤗 tokenizers, since the fast ones have native APIs to do
    that, we also return 🤗 `TokenSpan`s for compatibility with this API.
    """
    res: List[TokenSpan] = []
    pos = 0
    for length in word_lengths:
        while mask[pos] == special_tokens_code:
            pos += 1
        word_end = pos + length
        if any(token_type != sequence_tokens_code for token_type in mask[pos:word_end]):
            raise ValueError(
                "mask incompatible with tokenization:"
                f" needed {length} true tokens (1) at position {pos},"
                f" got {mask[pos:word_end]} instead"
            )
        res.append(TokenSpan(pos, word_end))
        pos = word_end

    return res


_T_BertLexer = TypeVar("_T_BertLexer", bound="BertLexer")


class BertLexer(torch.nn.Module):
    """
    This Lexer performs tokenization and embedding mapping with BERT
    style models. It concatenates a standard embedding with a BERT
    embedding.
    """

    def __init__(
        self,
        itos: Sequence[str],
        unk_word: str,
        embedding_size: int,
        word_dropout: float,
        bert_layers: Optional[Sequence[int]],
        bert_model: str,
        bert_subwords_reduction: Literal["first", "mean"],
        bert_weighted: bool,
        words_padding_idx: int,
    ):

        super().__init__()
        self.itos = itos
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.unk_word_idx = self.stoi[unk_word]

        try:
            self.bert = transformers.AutoModel.from_pretrained(
                bert_model, output_hidden_states=True
            )
        except OSError:
            config = transformers.AutoConfig.from_pretrained(bert_model)
            self.bert = transformers.AutoModel.from_config(config)

        self.bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
            bert_model, use_fast=True
        )
        # Shim for the weird idiosyncrasies of the RoBERTa tokenizer
        if isinstance(self.bert_tokenizer, transformers.GPT2TokenizerFast):
            self.bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
                bert_model, use_fast=True, add_prefix_space=True
            )

        self.embedding_size = embedding_size + self.bert.config.hidden_size

        self.embedding = torch.nn.Embedding(
            len(self.itos),
            embedding_size,
            padding_idx=words_padding_idx,
        )

        self.word_dropout = word_dropout
        self._dpout = 0.0

        # 🤗 has no unified API for the number of layers
        num_layers = next(
            n
            for param_name in ("num_layers", "n_layers", "num_hidden_layers")
            for n in [getattr(self.bert.config, param_name, None)]
            if n is not None
        )
        if bert_layers is None:
            bert_layers = list(range(num_layers))
        elif not all(
            -num_layers <= layer_idx < num_layers for layer_idx in bert_layers
        ):
            raise ValueError(
                f"Wrong BERT layer selections for a model with {num_layers} layers: {bert_layers}"
            )
        self.bert_layers = bert_layers
        # TODO: check if the value is allowed?
        self.bert_subwords_reduction = bert_subwords_reduction
        self.bert_weighted = bert_weighted
        self.layer_weights = torch.nn.Parameter(
            torch.ones(len(bert_layers), dtype=torch.float),
            requires_grad=self.bert_weighted,
        )
        self.layers_gamma = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float),
            requires_grad=self.bert_weighted,
        )
        self.out_dim = self.embedding_size + self.bert.config.hidden_size

    def train(self: _T_BertLexer, mode: bool = True) -> "_T_BertLexer":
        if mode:
            self._dpout = self.word_dropout
        else:
            self._dpout = 0.0
        return super().train(mode)

    def forward(self, inpt: BertLexerBatch) -> torch.Tensor:
        word_indices = inpt.word_indices
        if self._dpout:
            word_indices = integer_dropout(word_indices, self.unk_word_idx, self._dpout)
        word_embeddings = self.embedding(word_indices)

        bert_layers = self.bert(
            input_ids=inpt.bert_encoding["input_ids"], return_dict=True
        ).hidden_states
        # Shape: layers×batch×sequence×features
        selected_bert_layers = torch.stack(
            [bert_layers[i] for i in self.bert_layers], 0
        )

        if self.bert_weighted:
            # Torch has no equivalent to `np.average` so this is somewhat annoying
            # ! FIXME: recomputing the softmax for every batch is needed at train time but is wasting
            # ! time in eval
            # Shape: layers
            normal_weights = self.layer_weights.softmax(dim=0)
            # shape: batch×subwords_sequence×features
            bert_subword_embeddings = self.layers_gamma * torch.einsum(
                "l,lbsf->bsf", normal_weights, selected_bert_layers
            )
        else:
            bert_subword_embeddings = selected_bert_layers.mean(dim=0)
        # We already know the shape the BERT embeddings should have and we pad with zeros
        # shape: batch×sentence×features
        bert_embeddings = word_embeddings.new_zeros(
            (
                word_embeddings.shape[0],
                word_embeddings.shape[1],
                bert_subword_embeddings.shape[2],
            )
        )
        # FIXME: this loop is embarassingly parallel, there must be a way to parallelize it
        for sent_n, alignment in enumerate(inpt.subword_alignments):
            for word_n, span in enumerate(alignment, start=0):
                # shape: `span.end-span.start×features`
                bert_word_embeddings = bert_subword_embeddings[
                    sent_n, span.start : span.end, ...
                ]
                if self.bert_subwords_reduction == "first":
                    reduced_bert_word_embedding = bert_word_embeddings[0, ...]
                elif self.bert_subwords_reduction == "mean":
                    reduced_bert_word_embedding = bert_word_embeddings.mean(dim=0)
                else:
                    raise ValueError(
                        f"Unknown reduction {self.bert_subwords_reduction}"
                    )
                bert_embeddings[sent_n, word_n, ...] = reduced_bert_word_embedding

        return torch.cat((word_embeddings, bert_embeddings), dim=2)

    def make_batch(
        self,
        batch: Sequence[BertLexerSentence],
    ) -> BertLexerBatch:
        """Pad a batch of sentences."""
        words_batch, bert_batch, alignments = [], [], []
        for sent in batch:
            words_batch.append(sent.word_indices)
            bert_batch.append(sent.bert_encoding)
            alignments.append(sent.subwords_alignments)
        bert_encoding = self.bert_tokenizer.pad(bert_batch)
        return BertLexerBatch(
            bert_encoding=bert_encoding,
            padding_mask=pad_sequence(
                [torch.zeros(s.word_indices.shape[0], dtype=torch.bool) for s in batch],
                batch_first=True,
                padding_value=True,
            ),
            subword_alignments=alignments,
            word_indices=pad_sequence(
                words_batch, batch_first=True, padding_value=self.embedding.padding_idx
            ),
        )

    def encode(self, tokens_sequence: Sequence[str]) -> BertLexerSentence:
        """
        This maps word tokens to integer indexes.
        Args:
           tok_sequence: a sequence of strings
        """
        word_idxes = torch.tensor(
            [self.stoi.get(token, self.unk_word_idx) for token in tokens_sequence],
            dtype=torch.long,
        )

        # NOTE: for now the 🤗 tokenizer interface is not unified between fast and non-fast
        # tokenizers AND not all tokenizers support the fast mode, so we have to do this little
        # awkward dance. Eventually we should be able to remove the non-fast branch here.
        if self.bert_tokenizer.is_fast:
            bert_encoding = self.bert_tokenizer(
                tokens_sequence,
                is_split_into_words=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            # TODO: there might be a better way to do this?
            alignments = [
                bert_encoding.word_to_tokens(i) for i in range(len(tokens_sequence))
            ]
        else:
            bert_tokens = [
                self.bert_tokenizer.tokenize(token) for token in tokens_sequence
            ]
            bert_encoding = self.bert_tokenizer.encode_plus(
                [subtoken for token in bert_tokens for subtoken in token],
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            bert_word_lengths = [len(word) for word in bert_tokens]
            alignments = align_with_special_tokens(
                bert_word_lengths,
                bert_encoding["special_tokens_mask"],
            )

        return BertLexerSentence(word_idxes, bert_encoding, alignments)

    def save(self, model_path: pathlib.Path, save_weights: bool = True):
        model_path.mkdir(exist_ok=True, parents=True)
        with open(model_path / "config.yaml", "w") as out_stream:
            yaml.dump(
                {
                    "unk_word": self.unk_word,
                    "embedding_size": self.embedding_size,
                    "word_dropout": self.word_dropout,
                    "bert_layers": self.bert_layers,
                    "bert_model": self.bert_model,
                    "bert_subwords_reduction": self.bert_subwords_reduction,
                    "bert_weighted": self.bert_weighted,
                    "words_padding_idx": self.words_padding_idx,
                },
                out_stream,
            )
        with open(model_path / "lexicon.yaml", "w") as out_stream:
            yaml.dump(self.itos, out_stream)
        muppet_path = model_path / "muppet"
        self.bert.config.save_pretrained(str(muppet_path))
        self.bert_tokenizer.save(str(muppet_path))
        if save_weights:
            torch.save(self.state_dict(), model_path / "weights.pt")

    @classmethod
    def load(cls: Type[_T_BertLexer], model_path: pathlib.Path) -> _T_BertLexer:
        with open(model_path / "config.yaml") as in_stream:
            config = yaml.load(in_stream)
        with open(model_path / "lexicon.yaml") as in_stream:
            itos = yaml.load(in_stream)
        muppet_path = (model_path / "muppet").resolve()
        res = cls(itos=itos, bert_model=str(muppet_path), **config)
        weights_path = model_path / "weights.pt"
        if weights_path.exists():
            res.load_state_dict(torch.load(weights_path))
        return res
