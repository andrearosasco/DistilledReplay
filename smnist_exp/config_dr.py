import importlib
import os
from collections import OrderedDict

import torch
import numpy as np
from torchvision.transforms import transforms

model_config = OrderedDict([
    ('arch', 'lenet5'),
    ('n_classes', 10),
    ('input_shape', (1, 28, 28)),
])

data_config = OrderedDict([
    ('dataset', 'SplitMNIST'),
    ('valid', 0.0),
    ('num_workers', 4),
    ('train_transform', transforms.Compose([
        lambda x: np.array(x).reshape((1, 28, 28)),
        lambda x: np.pad(x, ((0, 0), (2, 2), (2, 2)), mode='minimum'),  # Padding is only required by LeNet
        lambda x: torch.FloatTensor(x),
        lambda x: x / 255.0,
        transforms.Normalize(np.array([0.1307]), np.array([0.3081]))
    ])),
    ('test_transform', transforms.Compose([
        lambda x: np.array(x).reshape((1, 28, 28)),
        lambda x: np.pad(x, ((0, 0), (2, 2), (2, 2)), mode='minimum'),  # Padding is only required by LeNet
        lambda x: torch.FloatTensor(x),
        lambda x: x / 255.0,
        transforms.Normalize(np.array([0.1307]), np.array([0.3081]))
    ])),
])


run_config = OrderedDict([
    ('experiment', 'run'),  # This configuration will be executed by run.py
    ('device', 'cuda'),
    ('tasks', [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), # , [4, 5], [6, 7], [8, 9]
    ('seed', 1234),
])

log_config = OrderedDict([
    ('wandb', True),
    ('wandb_name', 'DR'),
    ('print', True),
    ('images', True),  # Save the distilled images
])

param_config = OrderedDict([
    ('no_steps', 80),  # Training epoch performed by the model on the distilled dataset
    ('step', 'minibatch'),
    ('meta_lr', 0.1),  # Learning rate for distilling images
    ('model_lr', 0.05),  # Base learning rate for the model
    ('lr_lr', 0.0),  # Learning rate for the lrs of the model at each optimization step
    ('outer_steps', 70),  # Distillation epochsa
    ('inner_steps', 30),  # Optimization steps of the model
    ('batch_size', 128),  # Minibatch size used during distillation
    ('distill_batch_size', 128),
    ('buffer_size', 1),  # Number of examples per class kept in the buffer
])

config = OrderedDict([
    ('model_config', model_config),
    ('param_config', param_config),
    ('data_config', data_config),
    ('run_config', run_config),
    ('log_config', log_config),
])

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    experiment = importlib.import_module(config['run_config']['experiment'])
    experiment.run(config)