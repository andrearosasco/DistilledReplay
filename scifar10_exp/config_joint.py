import importlib
import os
from collections import OrderedDict

import numpy as np
from PIL import Image
from torchvision.transforms import transforms

model_config = OrderedDict([
    ('arch', 'lenet5'),
    ('n_classes', 10),
    ('input_shape', (3, 32, 32)),
])

data_config = OrderedDict([
    ('dataset', 'SplitCIFAR10'),
    ('valid', 0.0),
    ('num_workers', 4),
    ('train_transform', transforms.Compose([
        lambda x: Image.fromarray(x.reshape((3, 32, 32)).transpose((1, 2, 0))),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.5]), np.array([0.5]))])),
    ('test_transform', transforms.Compose([
        lambda x: Image.fromarray(x.reshape((3, 32, 32)).transpose((1, 2, 0))),
        transforms.ToTensor(),
        transforms.Normalize(np.array([0.5]), np.array([0.5]))
    ]))
])


run_config = OrderedDict([
    ('experiment', 'run'),  # This configuration will be executed by run.py
    ('device', 'cuda'),
    ('tasks', [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]), # , [4, 5], [6, 7], [8, 9]
    ('seed', 1234),
])

log_config = OrderedDict([
    ('wandb', True),
    ('wandb_name', 'joint'),
    ('print', True),
    ('images', True),  # Save the distilled images
])

param_config = OrderedDict([
    ('no_steps', 40),  # Training epoch performed by the model on the distilled dataset
    ('steps', 'minibatch'),  # epoch or minibatch('meta_lr', 0.1),  # Learning rate for distilling images
    ('meta_lr', 0.1),
    ('model_lr', 0.05),  # Base learning rate for the model
    ('lr_lr', 0.0),  # Learning rate for the lrs of the model at each optimization step
    ('outer_steps', 0),  # Distillation epochs
    ('inner_steps', 0),  # Optimization steps of the model
    ('batch_size', 1024),  # Minibatch size used during distillation
    ('distill_batch_size', 128),
    ('buffer_size', -1),  # Number of examples per class kept in the buffer
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