import yaml
from configs import cfg

# Load YAML file
with open('configs/audio_3dgs.yaml', 'r') as f:
    yaml_cfg = yaml.safe_load(f)

print("YAML parsed values:")
print(f"video type: {type(yaml_cfg['dataset']['video'])}")
print(f"video value: {yaml_cfg['dataset']['video']}")

print(f"\nDefault config video: {type(cfg.dataset.video)} = '{cfg.dataset.video}'")