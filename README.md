# DSLR
Data Science X Logistic Regression

### ToDo

- [x] [Describe](describe.py)
    - [x] get list of cleaned numerical dataset
    - [x] func(dataset: Series[float]) -> dic(min: float, max: float, ...)

- [x] Data Visualization
    - [x] [Histogram](histogram.py)
    - [x] [Scatter Plot](scatter_plot.py) (each pair of feature/course transfiguration/flying...)]
    - [x] [Pair Plot](pair_plot.py) (matrice of scatter plots with histograms on matrice diagonal)
    - [x] [Loss Plot](graph.py)

- [x] Logistic Regression [Formule markdown](/formula.md)
    - [x] [Logreg Train](logreg_train.py)
    - [ ] Optimizer
        - [x] exctract GD and SGD
        - [ ] run GD and SGD and compare plot
    - [x] Save weight, loss
    - [x] Refactor save weight, loss with pandas csv
    - [x] load weight
    - [x] [Logreg Predict](logreg_predict.py)

- [x] NaN Values to deal with
    - [x] Training or prediction of class1 >> use class1 mean/median value of feature
- [x] Normalize data with robust scaler scaled = (original - median) / (q3 - q1)
- [x] Data handling // preprossessing >> move cleaned and things like that in data
- [ ] Numerization (best hand / date of birth)
- [x] Unify output to csv for quartiles and weights: build a dataframe with desired shape, then write to csv
- [x] Calculate accuracy
- [ ] need 98% accuracy
- [x] Select % of each house examples and set as precision testing dataset

### Questions

what about the absence of data ?
during training and during prediction
fill with mean

During prediciton if multiple classes found for a given student how to decide which one?
Choose class with highest probability
