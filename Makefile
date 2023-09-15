start:
	jupyter lab --ip=0.0.0.0 --port=8000 --NotebookApp.allow_origin='*'

install:
	pip install -U pip setuptools -e .
