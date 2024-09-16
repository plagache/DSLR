# DSLR
Data Science X Logistic Regression

## ToDo

- [x] [Describe](describe.py)
    - [x] get list of cleaned numerical dataset
    - [x] func(dataset: Series[float]) -> dic(min: float, max: float, ...)
    - [x] add __main__
    - [x] join html string / avoid multiple write

- [x] Data Visualization
    - [x] [Histogram](histogram.py)
    - [x] [Scatter Plot](scatter_plot.py) (each pair of feature/course transfiguration/flying...)]
    - [x] [Pair Plot](pair_plot.py) (matrice of scatter plots with histograms on matrice diagonal)
    - [x] [Loss Plot](graph.py)
    - [x] add __main__
    - [x] extract string answers in variables.py
    - [x] rework --show in --web


- [x] Logistic Regression [Formule markdown](/formula.md)
    - [x] [Logreg Train](logreg_train.py)
    - [x] Optimizer
        - [x] exctract GD and SGD
        - [x] run GD and SGD and compare plot
    - [x] Save weight, loss
    - [x] Refactor save weight, loss with pandas csv
    - [x] load weight
    - [x] [Logreg Predict](logreg_predict.py)
    - [x] explicit the 2 function gradient_descente and Stochastique
    - [x] refacto test_sample/args.test_set into accuracy

- [x] NaN Values to deal with
    - [x] Training or prediction of class1 >> use class1 mean/median value of feature
- [x] Normalize data with robust scaler scaled = (original - median) / (q3 - q1)
- [x] Data handling // preprossessing >> move cleaned and things like that in data
- [x] Numerization (best hand / date of birth)
- [x] Unify output to csv for quartiles and weights: build a dataframe with desired shape, then write to csv
- [x] Calculate accuracy
- [x] need 98% accuracy
- [x] Select % of each house examples and set as precision testing dataset

- [x] split data process and calculation in predict
- [x] exctract accuracy
- [x] exctract sampling
- [x] unify data process of train and predict:
    - [x] robust scale with column name(q1,q2,q3,name)
    - [x] weight should match their column (w1 could be arithmancy or transfiguration but should be fixed)
    - [x] weight in the dataframe are alphabeticaly sorted and the resulting np array is in the same order
    - [x] exctract scaling from classer && courses list creation from Training
- [x] split loss formula to prevent log(0), only compute the y truth side.
- [x] split loss formula to prevent log(0), only compute the y truth side in SGD

- [ ] Web UI for selecting variables, training and display graph of accuracy, weight, loss
    - [x] refacto training / exctract data modification
    - [x] Gradio interface to selecte parameters / same as in variables.py
    - [x] button train with parameters
    - [x] display losses images
    - [x] create a group of tabs (train, predict)
    - [ ] defer rendering of predict tab
    - [ ] add a weights state and listen for changes of weights state to start predict rendering
    - [ ] add cross entropy tab
- [x] plot dataset_test with 5th color to determine classes
- [x] use scikit Logistic Regression to compare result on dataset_test
- [x] cross validation:
    - [x] refacto cross validation to work with new training prototype
- [x] modify set_quartiles to: set and get / refacto in consequence
- [x] use selected_features for brain creation
- [x] features selection / removing features can improve learning, by not having to learn an inefficient features
- [x] visualy understanding features selection with scatter plot


- [ ] organize files by theme:
    - [ ] Data Science -> describe, histogram, scatter, pair, flask ...
    - [ ] Logistic Regression -> train, predict, accuracy, sample, gradio ...

## Bonus

- cross_validation
- web
- describe per class
- SGD
- graph: loss/accuracies

## FAQ

what about the absence of data ?
during training and during prediction
fill with mean

During prediciton if multiple classes found for a given student how to decide which one?
Choose class with highest probability

Modify data :
	- remove outliers
	- data augmentation: good labeled data

L1, l2 regularization; not a fan of this one
