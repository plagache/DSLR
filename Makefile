VENV_PATH = .venv
BIN_PATH = ${VENV_PATH}/bin

env:
	python3.11 -m venv ${VENV_PATH}
	${BIN_PATH}/pip install -r requirement.txt

upgrade:
	${BIN_PATH}/pip install --upgrade pip

web: static
	# export FLASK_APP=homepage && export FLASK_ENV=development &&
	${BIN_PATH}/flask --app homepage.py run --host=0.0.0.0

debugweb: static
	${BIN_PATH}/flask --app homepage.py --debug run --host=0.0.0.0

extract:
	tar -xvf datasets.tgz

describe: extract
	${BIN_PATH}/python describe.py datasets/dataset_train.csv

webdescribe: extract
	${BIN_PATH}/python describe.py --web datasets/dataset_train.csv

scatter: extract static
	${BIN_PATH}/python scatter_plot.py --show datasets/dataset_train.csv

webscatter: extract static
	${BIN_PATH}/python scatter_plot.py datasets/dataset_train.csv

histogram: extract static
	${BIN_PATH}/python histogram.py --show datasets/dataset_train.csv

webhistogram: extract static
	${BIN_PATH}/python histogram.py datasets/dataset_train.csv

pair: extract static
	${BIN_PATH}/python pair_plot.py --show datasets/dataset_train.csv

webpair: extract static
	${BIN_PATH}/python pair_plot.py datasets/dataset_train.csv

static:
	mkdir -p static/Image/hist
	mkdir -p static/Image/scatter
	mkdir -p static/Image/pair

clean:
	rm -rf datasets
	rm -rf static/Image
	rm -rf templates/describe_table*.html

fclean: clean
	rm -rf ${VENV_PATH}

.SILENT: env clean fclean static histogram webhistogram scatter webscatter webdescribe describe extract web debugweb
.PHONY: env clean fclean static histogram webhistogram scatter webscatter webdescribe describe extract web debugweb
