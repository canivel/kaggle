# Download Competition Data

The Kaggle CLI needs API credentials. Steps:

1. Go to https://www.kaggle.com/settings → "API" section → "Create New Token"
2. This downloads `kaggle.json` — place it at `C:\Users\dcani\.kaggle\kaggle.json`
3. Then run:

```bash
python -m kaggle competitions download -c nvidia-nemotron-model-reasoning-challenge -p F:/kaggle/nvidia-nemotron-reasoning/data/
cd F:/kaggle/nvidia-nemotron-reasoning/data
unzip nvidia-nemotron-model-reasoning-challenge.zip
```

Or with PowerShell unzip:
```powershell
Expand-Archive -Path data\nvidia-nemotron-model-reasoning-challenge.zip -DestinationPath data\
```
