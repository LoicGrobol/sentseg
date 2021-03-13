import pathlib
from typing import Type, TypeVar

import torch
import torch.nn
import yaml

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

    def save(self, model_path: pathlib.Path):
        model_path.mkdir(exist_ok=True, parents=True)
        weights_file = model_path / "weights.pt"
        torch.save(self.state_dict(), weights_file)
        with open(model_path / "config.yaml", "w") as out_stream:
            yaml.dump(
                {
                    "depth": self.depth,
                    "n_heads": self.n_heads,
                },
                out_stream,
            )
        self.lexer.save(model_path / "lexer")

    @classmethod
    def load(cls: Type[_T_Segmenter], model_path: pathlib.Path) -> _T_Segmenter:
        with open(model_path / "config.yaml") as in_stream:
            config = yaml.load(in_stream)
        lexer = lexers.BertLexer.load(model_path / "lexer")
        return cls(lexer=lexer, **config)
