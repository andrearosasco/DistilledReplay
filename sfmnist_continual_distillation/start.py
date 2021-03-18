import os
import subprocess

root = '/home/rosasco/thesis/sfmnist_continual_distillation/'
configs = [root + 'config_naive_rev.py', root + 'config_joint_rev.py', root + 'config_sr_rev.py', root + 'config_dr_rev.py']

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    for c in configs:
        print(f'Executing {c}')
        proc = subprocess.Popen(['python', c])
        proc.wait()