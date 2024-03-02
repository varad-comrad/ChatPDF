import pathlib
import subprocess


path = pathlib.Path(__file__).parent / 'main.py'
subprocess.run(f'echo alias solus_env="source {path.parent.joinpath('env').joinpath('bin').joinpath('activate').absolute()}" >> ~/.bashrc', shell=True)
subprocess.run(f'echo alias solus="python {path.absolute()}" >> ~/.bashrc', shell=True)
