extract:
	tar -xvf datasets.tgz

describe: extract
	./describe.py datasets/dataset_train.csv

classe: extract
	./create_classes.py datasets/dataset_train.csv

clean:
	rm -rf datasets
