from __future__ import annotations

import logging
import os
import pathlib
import sys
import warnings

from typing import Any, cast, Dict, List, Literal, Optional, TextIO, Union

import click
import click_pathlib
import pydantic
import pytorch_lightning as pl
import transformers
import toml

from loguru import logger
from pytorch_lightning.utilities import rank_zero_only

from sentseg import data, lexers, segmenter
from sentseg.utils import smart_open


def setup_logging(
    verbose: bool, logfile: Optional[pathlib.Path] = None, replace_warnings: bool = True
):
    logger.remove(0)  # Remove the default logger
    if "SLURM_JOB_ID" in os.environ:
        appname = f"sentseg ({os.environ.get('SLURM_PROCID', 'somerank')} [{os.environ.get('SLURM_LOCALID', 'someproc')}@{os.environ.get('SLURMD_NODENAME', 'somenode')}])"
    else:
        appname = "sentseg"

    if verbose:
        log_level = "DEBUG"
        log_fmt = (
            f"[{appname}]"
            " <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
            " <level>{message}</level>"
        )
    else:
        logging.getLogger(None).setLevel(logging.CRITICAL)
        log_level = "INFO"
        log_fmt = (
            f"[{appname}]"
            " <green>{time:YYYY-MM-DD}T{time:HH:mm:ss}</green> {level}: "
            " <level>{message}</level>"
        )

    logger.add(
        sys.stderr,
        level=log_level,
        format=log_fmt,
        colorize=True,
    )

    if logfile:
        logger.add(
            logfile,
            level="DEBUG",
            format=(
                f"[{appname}] "
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
                "<level>{message}</level>"
            ),
            colorize=False,
        )

    # Deal with stdlib.logging

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Retrieve context where the logging call occurred, this happens to be in the 6th frame upward
            logger_opt = logger.opt(depth=6, exception=record.exc_info)
            logger_opt.log(record.levelno, record.getMessage())

    transformers_logger = logging.getLogger("transformers")
    # FIXME: ugly, but is there a better way?
    transformers_logger.handlers.pop()
    transformers_logger.addHandler(InterceptHandler())

    pl_logger = logging.getLogger("lightning")
    # FIXME: ugly, but is there a better way?
    pl_logger.handlers.pop()
    pl_logger.addHandler(InterceptHandler())

    # Deal with stdlib.warnings

    def showwarning(message, *args, **kwargs):
        logger.warning(message)

    if replace_warnings:
        warnings.showwarning = showwarning


class SavePretrainedModelCallback(pl.callbacks.Callback):
    def __init__(
        self,
        save_dir: pathlib.Path,
        period: int = 1,
    ):
        self.period = period
        self.save_dir = save_dir

    @rank_zero_only
    def on_epoch_end(
        self, trainer: pl.Trainer, pl_module: segmenter.SegmenterTrainModule
    ):
        if not trainer.current_epoch % self.period:
            epoch_save_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
            logger.info(f"Saving intermediate model to {epoch_save_dir}")
            pl_module.model.save(self.save_dir, save_weights=True)


@click.group()
def cli():
    pass


class SegmenterTrainingRunConfig(pydantic.BaseModel):
    max_epochs: int
    max_steps: Optional[int]


class SegmenterTrainingConfig(pydantic.BaseModel):
    run: SegmenterTrainingRunConfig
    hparams: segmenter.SegmenterTrainHparams


class Config(pydantic.BaseModel):
    segmenter: Dict[str, Any]
    training: SegmenterTrainingConfig


@cli.command(help="Train a segmenter model")
@click.argument(
    "config_path",
    type=click_pathlib.Path(resolve_path=True, dir_okay=False, exists=True),
)
@click.argument(
    "trainset_path",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
)
@click.option(
    "--accelerator",
    type=str,
    help="The lightning accelerator to use (see lightning doc)",
)
@click.option(
    "--device-batch-size",
    type=int,
    help=(
        "Number of samples in a processing batch"
        " (must be a divisor of training bath size, defaults to training batch size)"
    ),
)
@click.option(
    "--dev",
    "devset_path",
    type=click_pathlib.Path(resolve_path=True, exists=True, dir_okay=False),
    help="A CoNLL-U file for validation",
)
@click.option(
    "--guess-batch-size",
    is_flag=True,
    help=(
        "Try to find the max device batch size automatically"
        " (ignored if --device-batch-size is provided)"
    ),
)
@click.option(
    "--n-gpus",
    default=0,
    type=int,
    help="How many GPUs to train on. In ddp_cpu mode, this is the number of processes",
    show_default=True,
)
@click.option(
    "--n-nodes",
    type=int,
    default=os.environ.get("SLURM_JOB_NUM_NODES", 1),
    help="How many nodes to train on (for clusters), defaults to $SLURM_JOB_NUM_NODES if on SLURM and 1 otherwise",
)
@click.option(
    "--n-workers",
    type=int,
    default=0,
    help="How many data loading workers to use",
    show_default=True,
)
@click.option(
    "--out-dir",
    default=".",
    type=click_pathlib.Path(resolve_path=True, file_okay=False),
    help="Where to save the trained model, defaults to the current dir",
)
@click.option(
    "--save-period",
    type=int,
    help="The number of epoch between intermediate model saving, defaults to not saving intermediate models.",
    default=0,
)
@click.option(
    "--sharded-ddp",
    is_flag=True,
    help="Activate to use the sharded DDP mode (requires fairscale)",
)
@click.option("--profile", is_flag=True, help="Run in profiling mode")
@click.option("--verbose", is_flag=True, help="More detailed logs")
def train(
    accelerator: Optional[str],
    devset_path: Optional[pathlib.Path],
    config_path: pathlib.Path,
    device_batch_size: Optional[int],
    guess_batch_size: bool,
    n_gpus: int,
    n_nodes: int,
    n_workers: int,
    out_dir: pathlib.Path,
    profile: bool,
    save_period: int,
    sharded_ddp: bool,
    trainset_path: pathlib.Path,
    verbose: bool,
):
    if (slurm_procid := os.environ.get("SLURM_PROCID")) is not None:
        log_file = out_dir / "logs" / f"train{slurm_procid}.log"
    else:
        log_file = out_dir / "train.log"
    setup_logging(verbose, log_file)
    logger.debug(f"Current environment: {os.environ}")
    with open(config_path) as in_stream:
        config = Config.parse_obj(toml.load(in_stream))

    # NOTE: this is likely duplicated somewhere in pl codebase but we need it now unless pl rolls
    # out something like `optim_batch_size` that takes into account the number of tasks and the
    # number of samples per gpu
    if (num_slurm_tasks := os.environ.get("SLURM_NTASKS")) is not None:
        n_devices = int(num_slurm_tasks)
    elif n_gpus:
        n_devices = n_nodes * n_gpus
    else:
        n_devices = 1
    logger.info(f"Using {n_devices} devices.")

    vocab = data.vocab_from_conllu(trainset_path)
    lexer = lexers.BertLexer(vocab=vocab, **config.segmenter["lexer"])
    model = segmenter.Segmenter(lexer, **config.segmenter["model"])

    logger.info(f"Loading train dataset from {trainset_path}")
    train_set = data.SentDataset.from_conllu(trainset_path, segmenter=model)
    dev_set: Optional[data.TextDataset]
    if devset_path is not None:
        dev_set = data.SentDataset.from_conllu(devset_path, segmenter=model)
    else:
        dev_set = None

    logger.info("Creating training module")

    training_module = segmenter.SegmenterTrainModule(
        model, config=config.training.hparams
    )

    if device_batch_size is None:
        device_batch_size = training_module.config.batch_size
    elif training_module.config.batch_size < device_batch_size * n_devices:
        raise ValueError(
            f"Batch size ({training_module.config.batch_size}) is smaller than"
            f" loader batch size({device_batch_size} samples per device × {n_devices} devices)"
            " try using fewer devices"
        )
    elif training_module.config.batch_size % (device_batch_size * n_devices):
        remainder = training_module.config.batch_size % device_batch_size * n_devices
        logger.warning(
            f"Batch size ({training_module.config.batch_size}) is not a muliple"
            f" of loader batch size({device_batch_size} samples per device × {n_devices} devices)"
            f" the actual tuning batch size used will be {training_module.config.batch_size-remainder}."
        )

    # A pl Trainer batch is in fact one batch per device, so if we use multiple devices
    accumulate_grad_batches = training_module.config.batch_size // (
        device_batch_size * n_devices
    )

    # In DP mode, every batch is split between the devices
    if accelerator == "dp":
        loader_batch_size = device_batch_size * n_devices
    else:
        loader_batch_size = device_batch_size

    logger.info("Creating dataloaders")
    train_loader = data.SentLoader(
        train_set, batch_size=loader_batch_size, num_workers=n_workers, shuffle=True
    )

    val_loaders: Optional[List[data.TextLoader]]
    if dev_set is not None:
        val_loaders = [
            data.SentLoader(
                dev_set,
                batch_size=loader_batch_size,
                num_workers=n_workers,
                shuffle=False,
            )
        ]
    else:
        val_loaders = None

    logger.info("Creating trainer")
    additional_kwargs: Dict[str, Any] = dict()
    if profile:
        logger.info("Running in profile mode")
        profiler = pl.profiler.AdvancedProfiler(
            output_filename=str(out_dir / "profile.txt")
        )
        additional_kwargs.update({"profiler": profiler, "overfit_batches": 1024})

    if guess_batch_size:
        logger.info("Automatic batch size selection")
        additional_kwargs.update({"auto_scale_batch_size": "binsearch"})

    if accelerator == "ddp_cpu":
        # FIXME: works but seems like bad practice
        additional_kwargs["num_processes"] = n_gpus
        n_gpus = 0

    callbacks: List[pl.callbacks.Callback] = [
        pl.callbacks.ProgressBar(),
    ]
    if save_period:
        save_model(model, out_dir / "partway_models" / "initial")
        callbacks.append(
            SavePretrainedModelCallback(
                out_dir / "partway_models",
                save_period,
            )
        )

    if sharded_ddp:
        if accelerator == "ddp":
            logger.info("Using sharded DDP")
            cast(List[str], additional_kwargs.setdefault("plugins", [])).append(
                "ddp_sharded"
            )
        else:
            logger.warning(
                "--sharded-ddp only makes sense when using --accelerator=ddp. Ignoring the flag."
            )
    if n_gpus:
        logger.info(f"Training the model on {n_gpus} GPUs")
        logger.warning(
            "Half precision disabled since fast-transformers doesn't support it."
        )
        # additional_kwargs["precision"] = 16
    elif accelerator == "ddp_cpu":
        logger.info(
            f"Training the model on CPU in {additional_kwargs['num_processes']} processes"
        )
    else:
        logger.info("Training the model on CPU")

    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        default_root_dir=out_dir,
        accelerator=accelerator,
        gpus=n_gpus,
        limit_val_batches=1.0 if val_loaders is not None else 0,
        max_epochs=config.training.run.max_epochs,
        max_steps=config.training.run.max_steps,
        num_nodes=n_nodes,
        **additional_kwargs,
    )

    logger.info("Start training")

    trainer.fit(
        training_module, train_dataloader=train_loader, val_dataloaders=val_loaders
    )

    save_dir = out_dir / "model"
    save_model(model, save_dir)


@rank_zero_only
def save_model(
    model: transformers.PreTrainedModel,
    save_dir: pathlib.Path,
):
    """Save a segmenter model."""
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving model to {save_dir}")
    model.save(save_dir)


@cli.command(help="Segment a document using an existing model")
@click.argument(
    "model_path",
    type=click_pathlib.Path(resolve_path=True, file_okay=False, exists=True),
)
@click.argument(
    "input_path",
    type=click.Path(resolve_path=True, exists=True, dir_okay=False, allow_dash=True),
)
@click.argument(
    "output_path",
    type=click.Path(resolve_path=True, dir_okay=False, writable=True, allow_dash=True),
    default="-",
)
@click.option(
    "--from",
    "from_format",
    help="Format of the input file",
    type=click.Choice(["tokenized", "tsv"]),
    default="tokenized",
    show_default=True,
)
@click.option(
    "--to",
    "to_format",
    help="Format of the output file",
    type=click.Choice(["conll", "horizontal"]),
    default="conll",
    show_default=True,
)
def segment(
    from_format: Literal["tokenized", "tsv"],
    input_path: str,
    model_path: pathlib.Path,
    output_path: str,
    to_format: Literal["conll", "horizontal"],
):
    input_file: Union[str, TextIO]
    if input_path == "-":
        input_file = sys.stdin
    else:
        input_file = input_path
    model = segmenter.Segmenter.load(model_path)
    if from_format == "tokenized":
        with smart_open(input_file) as in_stream:
            words = in_stream.read().split()
            lines = None
    elif from_format == "tsv":
        with smart_open(input_file) as in_stream:
            lines = [line.strip() for line in in_stream]
            words = [line.split("\t")[0] for line in lines]
            # This is ugly but it works 😶
            if to_format == "horizontal":
                lines = None
    else:
        raise ValueError(f"Unknown format {from_format}")
    sents = model.segment(words, to_segment=lines)

    output_file: Union[str, TextIO]
    if output_path == "-":
        output_file = sys.stdout
    else:
        output_file = output_path

    if to_format == "conll":
        with smart_open(output_file, "w") as out_stream:
            for s in sents:
                for i, w in enumerate(s):
                    # This only works because our only input formats are tokenized and tsv
                    out_stream.write(f"{i}\t{w}\n")
                out_stream.write("\n")
    elif to_format == "horizontal":
        with smart_open(output_file, "w") as out_stream:
            for s in sents:
                out_stream.write(" ".join(w for w in s))
                out_stream.write("\n")
    else:
        raise ValueError(f"Unknown format {to_format}")


if __name__ == "__main__":
    cli()
