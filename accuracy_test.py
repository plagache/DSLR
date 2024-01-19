import argparse
from data_preprocessing import create_dataframe
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="A simple python program to test the accuracy of a given csv dataset")
parser.add_argument('samples', help='the test samples csv file')
parser.add_argument('prediction', help='the houses.csv file')
args = parser.parse_args()

samples = create_dataframe(args.samples)
prediction = create_dataframe(args.prediction)

y_true = samples["Hogwarts House"].tolist()
y_prediction = prediction["Hogwarts House"].tolist()

accuracy = accuracy_score(y_true, y_prediction)

print(f"accuracy: {accuracy * 100}%")
