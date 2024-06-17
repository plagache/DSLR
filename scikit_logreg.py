import argparse

import sklearn as sk
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline

import pandas


from data_preprocessing import create_dataframe

from variables import labels_column, selected_features, prediction_file

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument("train_set", help="the dataset csv file")
parser.add_argument("--test_set", action="store", help="use accuracy", dest="test_set", default=None)
args = parser.parse_args()


# scaler = RobustScaler()
scaler = StandardScaler()
estimator = SimpleImputer(strategy='mean')

train_sample = create_dataframe(args.train_set)
X_train = train_sample[selected_features]
X_train = estimator.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
print(X_train)
print(X_train.shape)
Y_train = train_sample["Hogwarts House"]

test_sample = create_dataframe(args.test_set)
X_test = test_sample[selected_features]
X_test = estimator.fit_transform(X_test)
X_test = scaler.fit_transform(X_test)
Y_test = test_sample["Hogwarts House"]

model = LogisticRegression(penalty=None, multi_class='ovr')
# model = SGDClassifier(loss='log_loss', penalty=None, max_iter=1000)
# model = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', penalty=None, max_iter=1000))
# model = make_pipeline(RobustScaler(), SGDClassifier(loss='log_loss', penalty=None, max_iter=1000))

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
Y_proba = model.predict_proba(X_test)
print(Y_proba)


prediction = pandas.DataFrame({"Hogwarts House": Y_pred})
prediction.to_csv(prediction_file, index_label="Index", header=[labels_column])
