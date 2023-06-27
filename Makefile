extract:
	tar -xvf datasets.tgz

describe: extract
	./describe.py datasets/dataset_train.csv

scatter: extract
	./scatter.py datasets/dataset_train.csv

clean:
	rm -rf datasets

.SILENT: clean scatter describe extract
.PHONY: clean scatter describe extract
