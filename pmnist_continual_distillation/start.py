import os
import subprocess

root = '/home/rosasco/thesis/pmnist_continual_distillation/'
configs = [root + 'config_joint.py', root + 'config_sr.py', root + 'config_dr.py']

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    for c in configs:
        print(f'Executing {c}')
        proc = subprocess.Popen(['python', c])
        proc.wait()