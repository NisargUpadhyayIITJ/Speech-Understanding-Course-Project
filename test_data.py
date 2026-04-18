import yaml
from datasets.asvspoof import ASVspoof2019Train, ASVspoof5Train, ASVspoof2019Dev, ASVspoof5Dev
from datasets.mlaad import MLAAD, MAILABS

with open("configs/train.yaml") as f:
    cfg = yaml.safe_load(f)

print("Testing ASVspoof19train...")
ds = ASVspoof2019Train(root_dir=cfg['data']['asvspoof2019_train']['root_dir'], meta_path=cfg['data']['asvspoof2019_train']['meta_path'])
print(f"Loaded {len(ds)} samples.")

print("Testing ASVspoof5train...")
ds = ASVspoof5Train(root_dir=cfg['data']['asvspoof5_train']['root_dir'], meta_path=cfg['data']['asvspoof5_train']['meta_path'])
print(f"Loaded {len(ds)} samples.")

print("Testing MLAAD...")
ds = MLAAD(root_dir=cfg['data']['mlaad']['root_dir'])
print(f"Loaded {len(ds)} samples.")

print("Testing MAILABS...")
ds = MAILABS(root_dir=cfg['data']['m_ailabs']['root_dir'])
print(f"Loaded {len(ds)} samples.")

print("Testing asvspoof19dev...")
ds = ASVspoof2019Dev(root_dir=cfg['data']['asvspoof2019_dev']['root_dir'], meta_path=cfg['data']['asvspoof2019_dev']['meta_path'])
print(f"Loaded {len(ds)} samples.")

print("Testing asvspoof5dev...")
ds = ASVspoof5Dev(root_dir=cfg['data']['asvspoof5_dev']['root_dir'], meta_path=cfg['data']['asvspoof5_dev']['meta_path'])
print(f"Loaded {len(ds)} samples.")

print("SUCCESS!")
