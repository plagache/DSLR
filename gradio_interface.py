import gradio as gr
import pandas as pd

from data_preprocessing import (
    create_classes,
    create_labels,
    create_training_data,
    robust_scale,
    select_numerical_features,
)
from graph import draw_graphs
from logreg_predict import predict
from logreg_train import training
from nn import Brain
from variables import labels_column, learning_rate, learning_rate_decay, scheduler_type, selected_features, steps, stochastic

# Load the dataset
dataset_train = pd.read_csv("datasets/dataset_train.csv")
numerical_features = select_numerical_features(dataset_train)
dataset_test = pd.read_csv("datasets/dataset_test.csv")
learned_weights = pd.read_csv("tmp/weights.csv")
learned_quartiles = pd.read_csv("tmp/quartiles.csv")


# Function to filter dataset based on selected features
def filter_dataset(selected_features):
    if selected_features:
        filtered_data = dataset_train[selected_features]
        return filtered_data
    else:
        return "No features selected"


def toggle_panel(stochastic):
    return gr.update(visible=stochastic)


def gradio_train(selected_features, learning_rate, steps, stochastic, learning_rate_decay, scheduler):
    if len(selected_features) == 0:
        return "No features selected"
    else:
        scheduler = scheduler
        dataset_selected = dataset_train[selected_features]
        x_train = create_training_data(dataset_selected)
        features = x_train.columns.tolist()
        features_tensor = x_train.to_numpy()

        classes = create_classes(dataset_train)
        labels = create_labels(dataset_train, classes)
        labels_tensor = labels.to_numpy().T

        brain = Brain(classes, features)

        test_sample = pd.DataFrame(dataset_test)
        numerical_features = select_numerical_features(test_sample)
        selected_features = numerical_features[selected_features]
        x_test = create_training_data(selected_features)
        losses, weights, accuracies = training(brain, features_tensor, labels_tensor, learning_rate, steps, stochastic, x_test, test_sample)
        losses_df = pd.DataFrame(losses, columns=classes)
        accuracy_df = pd.DataFrame(accuracies)
        figure = draw_graphs(losses_df, classes, accuracy_df)
        final_accuracy = f"{accuracies[-1] * 100}%"
        return figure, final_accuracy, selected_features


def gradio_predict(selected_features):
    assert learned_quartiles is not None, "Error: CSV file 'tmp/quartiles.csv' could not be read"
    assert learned_weights is not None, "Error: CSV file 'tmp/weights.csv' could not be read"
    if selected_features is None:
        return "No features selected"

    dataset = dataset_test
    dataset = dataset.select_dtypes(include=["float64"])
    dataset = dataset[selected_features.columns]

    parameters = learned_weights
    quartiles = learned_quartiles

    list_quartiles = list(quartiles.itertuples(index=False, name=None))

    scaleddataset = robust_scale(dataset, list_quartiles).to_numpy()

    classes = parameters[labels_column].tolist()
    parameters = parameters.set_index(labels_column)
    features = parameters.columns.tolist()

    brain = Brain(classes, features, weights=parameters.to_numpy())

    prediction = predict(brain, scaleddataset)
    return prediction


with gr.Blocks() as demo:
    selected_features_state = gr.State()
    with gr.Tab("Train"):
        with gr.Row():
            with gr.Column():
                selected_features = gr.CheckboxGroup(choices=list(numerical_features.columns), label="Select Features", value=selected_features)

                stochastic = gr.Checkbox(value=stochastic, label="stochastic")
                with gr.Row(visible=False) as panel:
                    learning_rate_decay = gr.Number(value=learning_rate_decay, minimum=1e-7, maximum=10, label="learning rate decay")
                    scheduler = gr.Radio(choices=scheduler_type, label="Scheduler")

                # Update the panel visibility based on the checkbox
                stochastic.change(toggle_panel, inputs=stochastic, outputs=panel)

                learning_rate = gr.Number(value=learning_rate, minimum=1e-7, maximum=10, label="learning rate")
                steps = gr.Number(value=steps, minimum=1, maximum=4000, label="steps")
                train_button = gr.Button(value="Train")
            with gr.Column():
                accuracy = gr.Textbox(label="Final Accuracy", interactive=False)
                figure = gr.Plot(label="Losses")

        train_button.click(
            fn=gradio_train,
            inputs=[selected_features, learning_rate, steps, stochastic, learning_rate_decay, scheduler],
            outputs=[figure, accuracy, selected_features_state],
        )
    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column():
                prediction = gr.DataFrame(label="Prediction")
                predict_button = gr.Button(value="Predict")

            predict_button.click(fn=gradio_predict, inputs=[selected_features_state], outputs=[prediction])


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=False)
