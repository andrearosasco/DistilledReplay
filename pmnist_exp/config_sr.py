import importlib
import os
from collections import OrderedDict

import torch
from torchvision.transforms import transforms

model_config = OrderedDict([
    ('arch', 'mlp2'),
    ('n_classes', 10),
    ('dropout', 0.5)
])

data_config = OrderedDict([
    ('dataset', 'PermutedMNIST'),
    ('valid', 0.0),
    ('num_workers', 4),
    ('train_transform', transforms.Compose([
            lambda x: torch.FloatTensor(x),
            lambda x: x / 255.0,
            lambda x: (x - 0.1307) / 0.3081,
        ])),
    ('test_transform', transforms.Compose([
            lambda x: torch.FloatTensor(x),
            lambda x: x / 255.0,
            lambda x: (x - 0.1307) / 0.3081,
        ]))
])


run_config = OrderedDict([
    ('experiment', 'run'),  # This configuration will be executed by distill.py
    ('device', 'cuda'),
    ('tasks', list(range(10))),  # , [4, 5], [6, 7], [8, 9]
    ('save', 'task1.distilled'),  # Path for the distilled dataset
    ('seed', 1234),
])

log_config = OrderedDict([
    ('wandb', True),
    ('wandb_name', 'SR'),
    ('print', True),
    ('images', True),  # Save the distilled images
])

param_config = OrderedDict([
    ('no_steps', 3),  # Training epoch performed by the model on the distilled dataset
    ('steps', 'epoch'),  # epoch or minibatch
    ('meta_lr', 0.1),  # Learning rate for distilling images
    ('model_lr', 0.05),  # Base learning rate for the model
    ('lr_lr', 0.0),  # Learning rate for the lrs of the model at each optimization step
    ('outer_steps', 0),  # Distillation epochs
    ('inner_steps', 0),  # Optimization steps of the model
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    experiment = importlib.import_module(config['run_config']['experiment'])
    experiment.run(config)