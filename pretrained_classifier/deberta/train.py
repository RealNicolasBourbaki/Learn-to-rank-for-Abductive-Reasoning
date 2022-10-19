__credits__ = '{https://github.com/isi-nlp/ai2}'
# Reference: https://pytorch-lightning.readthedocs.io/en/latest/

import torch
import random
import numpy as np
from pytorch_lightning import Trainer
from loguru import logger
from model import Classifier
import omegaconf


def train(config):
    logger.info(config)

    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.backends.cuda.deterministic = True
        torch.backends.cuda.benchmark = False

    model = Classifier(config)
    trainer = Trainer(
        gradient_clip_val=0,
        num_nodes=1,
        gpus=None if not torch.cuda.is_available() else 1,
        log_gpu_memory=True,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        max_epochs=config["max_epochs"],
        min_epochs=1,
        val_check_interval=0.1,
        flush_logs_every_n_steps=100,
        log_every_n_steps=10,
        distributed_backend="ddp",
        weights_summary='top',
        num_sanity_val_steps=5,
        resume_from_checkpoint=None,
    )
    trainer.fit(model)

    pass


if __name__ == "__main__":
    config = omegaconf.OmegaConf.load("config-deberta.yaml")
    train(config)
