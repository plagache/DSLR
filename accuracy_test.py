import argparse
from data_preprocessing import create_dataframe
from sklearn.metrics import accuracy_score
from variables import labels_column

def test_accuracy(samples, prediction, labels_column):
    y_true = samples[labels_column].tolist()
    y_prediction = prediction[labels_column].tolist()

    accuracy = accuracy_score(y_true, y_prediction)
    return accuracy

if __name__ == "__main__":
    print("\n------------ Accuracy -----------")
    parser = argparse.ArgumentParser(description="A simple python program to test the accuracy of a given csv dataset")
    parser.add_argument('samples', help='the test samples csv file')
    parser.add_argument('prediction', help='the houses.csv file')
    args = parser.parse_args()

    samples = create_dataframe(args.samples)
    prediction = create_dataframe(args.prediction)
    accuracy = test_accuracy(samples, prediction, labels_column)
    print(f"\naccuracy: {accuracy * 100}%")
