VENV_PATH = .venv
BIN_PATH = ${VENV_PATH}/bin
TRAIN_SET = datasets/dataset_train.csv
TEST_SET = datasets/dataset_test.csv

setup: env pip_upgrade install

env:
	uv venv --python 3.11
	ln -sf ${BIN_PATH}/activate activate

pip_upgrade:
	uv pip install --upgrade pip

install:
	uv pip install -r requirements.txt

upgrade:
	uv pip install -r requirements.txt --upgrade

list:
	${BIN_PATH}/pip list

size:
	du -hd 0
	du -hd 0 ${VENV}

extract:
	tar -xf datasets.tgz

###### DS

describe: extract
	${BIN_PATH}/python describe.py ${TRAIN_SET}

histogram: extract static
	${BIN_PATH}/python histogram.py ${TRAIN_SET}

scatter: extract static
	${BIN_PATH}/python scatter_plot.py ${TRAIN_SET}

pair: extract static
	${BIN_PATH}/python pair_plot.py ${TRAIN_SET}

###### LR

train: extract
	${BIN_PATH}/python logreg_train.py ${TRAIN_SET}

predict: tmp/weights.csv tmp/quartiles.csv
	${BIN_PATH}/python logreg_predict.py ${TEST_SET} tmp/weights.csv tmp/quartiles.csv

repredict: extract predict

###### UI

web: static
	# export FLASK_APP=homepage && export FLASK_ENV=development &&
	${BIN_PATH}/flask --app homepage.py run --host=0.0.0.0

debugweb: static
	${BIN_PATH}/flask --app homepage.py --debug run --host=0.0.0.0

webdescribe: extract
	${BIN_PATH}/python describe.py --web ${TRAIN_SET}

webhistogram: extract static
	${BIN_PATH}/python histogram.py --web ${TRAIN_SET}

webscatter: extract static
	${BIN_PATH}/python scatter_plot.py --web ${TRAIN_SET}

webpair: extract static
	${BIN_PATH}/python pair_plot.py --web ${TRAIN_SET}

gradio_interface: extract static
	${BIN_PATH}/gradio gradio_interface.py
	
###### Accuracy

sample: static extract
	${BIN_PATH}/python sampler.py ${TRAIN_SET}

trainacc:
	${BIN_PATH}/python logreg_train.py ${TRAIN_SET} --accuracy ${TEST_SET}

accuracy:
	${BIN_PATH}/python accuracy_test.py ${TEST_SET} houses.csv

fullaccuracy: sample trainacc predict accuracy

graph: static
	${BIN_PATH}/python graph.py tmp/losses.csv tmp/accuracies.csv

###### Test

concat:
	${BIN_PATH}/python concat_dataset.py ${TRAIN_SET} ${TEST_SET}

histogram_test: extract static concat
	${BIN_PATH}/python histogram.py ${TRAIN_SET}

scatter_test: extract static concat
	${BIN_PATH}/python scatter_plot.py ${TRAIN_SET}

pair_test: extract static concat
	${BIN_PATH}/python pair_plot.py ${TRAIN_SET}

###### Scikit

scikit_logreg: extract static
	${BIN_PATH}/python scikit_logreg.py ${TRAIN_SET} --test_set ${TEST_SET}

diff_scikit:
	diff -y --suppress-common-lines houses.csv houses_scikit.csv | wc -l

###### Cross Validation

cross_validation: extract static
	${BIN_PATH}/python cross_validation.py ${TRAIN_SET}


###### Houskeeping

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
	rm -rf houses_scikit.csv

fclean: clean
	rm -rf ${VENV_PATH}
	rm -f activate

.SILENT:
.PHONY: setup env pip_upgrade install upgrade list size extract clean fclean static \
	webdescribe describe histogram webhistogram scatter webscatter pair webpair web \
	concat histogram_test scatter_test pair_test \
	debugweb sample trainacc accuracy fullaccuracy graph
