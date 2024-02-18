import numpy as np
import pandas
from logreg_predict import predict
from nn import Brain
from optim import sgd, gd
from data_preprocessing import create_dataframe, create_classes, create_training_data, create_labels
import argparse
from tqdm import tqdm
from variables import labels_column, learning_rate, steps
from accuracy_test import test_accuracy
from sampler import split_dataframe

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--web', action='store_true', help='export data to html output')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

print("\n------------ Training -----------")

weights_matrix = []
losses_matrix = []

# we want him to use dataset test either samples or not
# test_sample, train_sample = split_dataframe(dataset, 0)
test_sample, train_sample = dataset, dataset
dataset = train_sample
samples = create_training_data(test_sample)

x_train = create_training_data(dataset)
features = x_train.columns.tolist()
features_tensor = x_train.to_numpy()

classes = create_classes(dataset)
labels = create_labels(dataset, classes)
labels_tensor = labels.to_numpy().T

brain = Brain(classes, features)

# losses = np.zeros((1,4))
losses = []
learning_rate = learning_rate
steps = steps
for step in (t:=tqdm(range(steps))):
    # loss = sgd(features_tensor, neuron, class_labels, learning_rate)
    loss = gd(brain, learning_rate, features_tensor, labels_tensor)

    prediction_test = predict(brain, samples)
    accuracy = test_accuracy(test_sample, prediction_test, labels_column)

    # print(loss, losses)
    # print(np.append(losses, loss))
    losses.append(loss)

    t.set_description(f"accuracy: {accuracy * 100:.2f}%")
    # t.set_description(f"loss:{losses[-1]:.6f}| accuracy:{accuracy * 100:.2f}%| step")


print(losses)
losses = np.concatenate(losses, axis=0)
print(losses)
# losses_matrix = np.array(losses_matrix)
# losses_df = pandas.DataFrame(losses_matrix.T, columns=classes)
# losses_df.to_csv("tmp/losses.csv", index=False)

# weights_df = pandas.DataFrame(weights_matrix, index=classes, columns=features)
weights_df = pandas.DataFrame(brain.weights, index=classes, columns=features)
weights_df.to_csv("tmp/weights.csv", index_label=labels_column)
