import argparse

import pandas
import sklearn as sk
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from data_preprocessing import create_dataframe
from variables import labels_column

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument("train_set", help="the dataset csv file")
parser.add_argument("--test_set", action="store", help="use accuracy", dest="test_set", default=None)
args = parser.parse_args()

prediction_file = "houses_scikit.csv"

features = [
    "Ancient Runes",
    "Arithmancy",
    "Astronomy",
    "Care of Magical Creatures",
    "Charms",
    "Defense Against the Dark Arts",
    "Divination",
    "Flying",
    "Herbology",
    "History of Magic",
    "Muggle Studies",
    "Potions",
    "Transfiguration",
]

number_of_feature = 6
# scaler = RobustScaler()
scaler = StandardScaler()
estimator = SimpleImputer(strategy="median")
selector = SelectKBest(f_classif, k=number_of_feature)

train_sample = create_dataframe(args.train_set)
X_train = train_sample[features]
train_columns = X_train.columns
X_train = estimator.fit_transform(X_train)
X_train = scaler.fit_transform(X_train)
Y_train = train_sample["Hogwarts House"]
X_new = selector.fit(X_train, Y_train)
X_new = selector.fit_transform(X_train, Y_train)
mask = selector.get_support()
mask = selector.get_support(indices=True)
selected_features = train_columns[mask]
f_statistic, p_values = f_classif(X_train, Y_train)
print(mask)
print(selected_features)
print(f_statistic)
print(p_values)
print(type(f_statistic))
print(type(p_values))

test_sample = create_dataframe(args.test_set)
# X_test = test_sample[selected_features]
X_test = test_sample[selected_features]
X_test = estimator.fit_transform(X_test)
X_test = scaler.fit_transform(X_test)
Y_test = test_sample["Hogwarts House"]

classifier = OneVsRestClassifier(LogisticRegression()).fit(X_new, Y_train)
class_prediction = classifier.predict(X_test)
print(class_prediction)
class_pred = pandas.DataFrame({"Hogwarts House": class_prediction})
# print(class_pred)


model = LogisticRegression(penalty=None, multi_class="ovr")
# model = SGDClassifier(loss='log_loss', penalty=None, max_iter=1000)
# model = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', penalty=None, max_iter=1000))
# model = make_pipeline(RobustScaler(), SGDClassifier(loss='log_loss', penalty=None, max_iter=1000))

# model.fit(X_train, Y_train)
model.fit(X_new, Y_train)

Y_pred = model.predict(X_test)
Y_proba = model.predict_proba(X_test)
print(Y_pred)
print(type(Y_pred))
print((Y_pred == class_prediction).all())
# print(Y_proba)


# prediction = pandas.DataFrame({"Hogwarts House": Y_pred})
# print(prediction)
# prediction.to_csv(prediction_file, index_label="Index", header=[labels_column])
