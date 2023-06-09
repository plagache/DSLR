env:
	python -m venv .
	./bin/pip3.11 install -r requirement.txt

upgrade:
	./bin/pip3.11 install --upgrade pip

web: static
	export FLASK_APP=homepage && export FLASK_ENV=development && ./bin/flask run --host=0.0.0.0

extract:
	tar -xvf datasets.tgz

describe: extract
	./bin/python3.11 describe.py datasets/dataset_train.csv

scatter: extract static
	./bin/python3.11 scatter.py datasets/dataset_train.csv

histogram: extract static
	./bin/python3.11 histogram.py --show datasets/dataset_train.csv

webhistogram: extract static
	./bin/python3.11 histogram.py datasets/dataset_train.csv

static:
	mkdir -p static/Image/hist
	mkdir -p static/Image/scatter
	mkdir -p static/Image/pair

clean:
	rm -rf datasets
	rm -rf static/Image

.SILENT: env clean static histogram scatter describe extract web
.PHONY: env clean static histogram scatter describe extract web
