import gradio as gr
import pandas as pd

from data_preprocessing import (
    create_classes,
    create_labels,
    create_training_data,
    select_numerical_features,
)
from graph import draw_graphs
from logreg_train import training
from nn import Brain


# Function to select only the features list
def load_features(data):
    return


# Load the dataset
dataset_train = pd.read_csv("datasets/dataset_train.csv")
numerical_features = select_numerical_features(dataset_train)
dataset_test = pd.read_csv("datasets/dataset_test.csv")


# Function to filter dataset based on selected features
def filter_dataset(selected_features):
    if selected_features:
        filtered_data = dataset_train[selected_features]
        return filtered_data
    else:
        return "No features selected"


def gradio_train(selected_features, learning_rate, steps, stochastic):
    if len(selected_features) == 0:
        return "No features selected"
    else:
        dataset_selected = dataset_train[selected_features]
        x_train = create_training_data(dataset_selected)
        features = x_train.columns.tolist()
        features_tensor = x_train.to_numpy()

        classes = create_classes(dataset_train)
        labels = create_labels(dataset_train, classes)
        labels_tensor = labels.to_numpy().T

        brain = Brain(classes, features)

        test_sample = dataset_test[selected_features]
        x_test = create_training_data(selected_features)
        losses, weights, accuracies = training(brain, features_tensor, labels_tensor, learning_rate, steps, stochastic, test_sample)
        return draw_graphs(losses, classes, accuracies)


def load_code():
    with open("variables.py", "r") as file:
        code = file.read()
    return code


def save_code(new_code):
    with open("variables.py", "w") as file:
        file.write(new_code)
    return "variables.py updated successfully!"


with gr.Blocks() as demo:
    with gr.Tab("Train"):
        with gr.Row():
            with gr.Column():
                # selected_features = gr.Interface(
                #     fn=filter_dataset,
                #     inputs=gr.CheckboxGroup(choices=list(dataset_train.columns), label="Select Features"),
                #     outputs=gr.DataFrame(label="Filtered Dataset"),
                # )
                selected_features = gr.CheckboxGroup(choices=list(numerical_features.columns), label="Select Features")
                stochastic = gr.Checkbox(value=False, label="stochastic")
                learning_rate = gr.Number(value=0.2, label="learning_rate")
                steps = gr.Number(value=2000, label="steps")
                submit_btn = gr.Button(value="Train")
            with gr.Column():
                result = gr.Plot()

        submit_btn.click(gradio_train, inputs=[selected_features, learning_rate, steps, stochastic], outputs=[result], api_name=False)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=False)
