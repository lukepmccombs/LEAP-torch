import torch
from toolz import curry
import os

from leap_ec.util import get_step
from leap_ec.ops import listlist_op
from leap_ec.global_vars import context


@curry
@listlist_op
def save_best_probe(population, *, save_dir, modulo=1, context=context):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    step = get_step(context)
    if step % modulo == 0:
        best_ind = max(population)
        torch.save(best_ind.genome, os.path.join(save_dir, f"best-step{step}.pt"))

    return population


