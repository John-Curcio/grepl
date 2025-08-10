openmmlab:
	poetry run python -m pip install -U pip setuptools wheel
	poetry run pip install "mmengine>=0.10,<1.0"
	poetry run pip install "mmcv==2.2.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/index.html
	poetry run pip install "mmpose>=1.3.2,<2"
