from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
data_path = f'{ROOT_DIR}/.data'
cache_path = f'{ROOT_DIR}/.cache'
checkpoint_path = f'{cache_path}/checkpoints'
results_dir_path = f'{ROOT_DIR}/results'

unk_token = '###'