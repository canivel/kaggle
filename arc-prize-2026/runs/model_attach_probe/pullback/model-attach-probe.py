import os
from pathlib import Path

root = Path('/kaggle/input')
print('MODEL-ATTACH-PROBE listing of /kaggle/input:')
for dirpath, dirnames, filenames in os.walk(root):
    depth = len(Path(dirpath).relative_to(root).parts)
    if depth <= 4:
        print(' ' * depth * 2 + Path(dirpath).name + '/')
    if depth >= 4:
        dirnames.clear()
hits = [p for p in root.rglob('config.json')]
print('config.json hits:', [str(p) for p in hits[:10]])
