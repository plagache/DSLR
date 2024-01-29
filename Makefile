VENV_PATH = .venv
BIN_PATH = ${VENV_PATH}/bin
TRAIN_SET = datasets/dataset_train.csv
TEST_SET = datasets/dataset_test.csv

env:
	python3.11 -m venv ${VENV_PATH}
	${BIN_PATH}/pip install -r requirement.txt
	ln -sf ${BIN_PATH}/activate activate

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
	${BIN_PATH}/python describe.py ${TRAIN_SET}

sample: static extract
	${BIN_PATH}/python sampler.py ${TRAIN_SET}

train: static
	${BIN_PATH}/python logreg_train.py ${TRAIN_SET}

graph: extract static
	${BIN_PATH}/python graph.py tmp/losses.csv

predict:
	${BIN_PATH}/python logreg_predict.py ${TEST_SET} tmp/weights.csv tmp/quartiles.csv

accuracy:
	${BIN_PATH}/python accuracy_test.py ${TEST_SET} houses.csv

fullaccuracy: sample train predict accuracy
	

webdescribe: extract
	${BIN_PATH}/python describe.py --web ${TRAIN_SET}

scatter: extract static
	${BIN_PATH}/python scatter_plot.py --show ${TRAIN_SET}

webscatter: extract static
	${BIN_PATH}/python scatter_plot.py ${TRAIN_SET}

histogram: extract static
	${BIN_PATH}/python histogram.py --show ${TRAIN_SET}

webhistogram: extract static
	${BIN_PATH}/python histogram.py ${TRAIN_SET}

pair: extract static
	${BIN_PATH}/python pair_plot.py --show ${TRAIN_SET}

webpair: extract static
	${BIN_PATH}/python pair_plot.py ${TRAIN_SET}

static:
	mkdir -p static/Image/hist
	mkdir -p static/Image/scatter
	mkdir -p static/Image/pair
	mkdir -p static/Image/loss

clean:
	rm -rf datasets
	rm -rf static/Image
	rm -rf templates/describe_table*.html
	rm -rf tmp/*
	rm -rf houses.csv

fclean: clean
	rm -rf ${VENV_PATH}
	rm -f activate

.SILENT:
.PHONY: env clean fclean static histogram webhistogram scatter webscatter webdescribe describe extract web debugweb
