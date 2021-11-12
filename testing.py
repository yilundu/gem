'''Implements a generic training loop.
'''

import torch
import summaries
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import collections

def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def test(model, test_dataloader, loss_fn, output_fn, checkpoint_path, model_eval_fn=None, num_forward_per_sample=1):
    assert checkpoint_path is not None, "Have to pass in a checkpoint path!"
    model.load_state_dict(torch.load(checkpoint_path))

    if model_eval_fn is None:
        model_eval_fn = model.forward

    with torch.no_grad():
        with tqdm(total=len(test_dataloader)) as pbar:
            for step, (model_input, gt) in enumerate(test_dataloader):
                for _ in range(num_forward_per_sample):
                    model_input = dict_to_gpu(model_input)
                    gt = dict_to_gpu(gt)

                    model_output = model_eval_fn(model_input)
                    losses = loss_fn(model_output, gt)
                    output_fn(model_output, gt, losses)
                    pbar.update(1)
