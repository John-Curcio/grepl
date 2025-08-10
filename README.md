# README

Ridiculous

# install

```
poetry install
make openmmlab
```

# sanity check successful installation of mmpose, mmcv stuff

Need `mmpose` for object detection and pose estimation. But setup seems incompatible with poetry.

```
poetry run python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
import mmcv, mmengine, mmpose
print("mmcv:", mmcv.__version__)
print("mmengine:", mmengine.__version__)
print("mmpose:", mmpose.__version__)
PY
```