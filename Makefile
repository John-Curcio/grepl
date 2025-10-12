openmmlab:
	uv pip install --upgrade pip setuptools wheel
	uv pip install "mmengine>=0.10,<1.0"
	uv pip install "mmcv==2.2.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/index.html
	uv pip install "mmpose>=1.3.2,<2"
