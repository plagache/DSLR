# DSLR
Data Science X Logistic Regression

### ToDo

- [x] Describe
    - [x] get list of cleaned numerical dataset
    - [x] func(dataset: Series[float]) -> dic(min: float, max: float, ...)

- [x] Data Visualization
    - [x] Histogram
    - [x] Scatter Plot (each pair of feature/course transfiguration/flying...)
    - [x] Pair Plot (matrice of scatter plots with histograms on matrice diagonal)
    - [ ] Plot the loss evolution

- [x] Logistic Regression [Formule markdown](/formula.md)
    - [x] train
        - [ ] SGD
    - [x] Save weight, loss
    - [x] load weight
    - [x] predict

- [x] NaN Values to deal with
    - [x] Training or prediction of class1 >> use class1 mean/median value of feature
- [x] Normalize data with robust scaler scaled = (original - median) / (q3 - q1)
- [x] Data handling // preprossessing >> move cleaned and things like that in data
- [ ] Numerization (best hand / date of birth)

### Questions

what about the absence of data ?
during training and during prediction
fill with mean

During prediciton if multiple classes found for a given student how to decide which one?
Choose class with highest probability
