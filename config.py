from pathlib import Path

# ROOT_DIR = Path(__file__).parent.parent
ROOT_DIR = 'proot'
RAID_PATH = 'praid'
data_path = f'{RAID_PATH}/.data'
cache_path = f'{RAID_PATH}/.cache'
checkpoint_path = f'{cache_path}/checkpoints'
results_dir_path = f'{ROOT_DIR}/results'

unk_token = '###'