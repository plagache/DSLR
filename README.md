# DSLR
Data Science X Logistic Regression

## Installation and dependencies
You need python, [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#uv), make
```bash
git clone https://github.com/plagache/DSLR
cd DSLR
make setup
```

## Usage

### Mandatory part

```bash
# display information for all numerical features of a given dataset
make describe

# display respectively a histogram plot, a scatter plot and a pair plot
make histogram
make scatter
make pair

# train our model with gradient descent
make train

# generate a prediction file houses.csv
make predict
```

### Bonus part

```bash
# launch a webserver that allow to display all the plot nicely
make debugweb

# launch a gradio server to tune our training easily
make gradio_interface

# implementation with scikit, for comparaison only
make scikit_logreg

# train with k-fold cross validation
make cross_validation
```
