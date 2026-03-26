import shutil
from pathlib import Path
from datetime import datetime

base_config = Path("config").resolve()
snapshot_root = Path("runs/config_snapshots").resolve()
snapshot_root.mkdir(parents=True, exist_ok=True)

job_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
snapshot_dir = snapshot_root / job_id / "config"

shutil.copytree(base_config, snapshot_dir)

print("Snapshot created at:")
print(snapshot_dir)