import gradio as gr
import pandas as pd

from data_preprocessing import (
    create_classes,
    create_labels,
    create_training_data,
    robust_scale,
    select_numerical_features,
    get_selected_features
)
from graph import draw_graphs
from logreg_predict import predict
from logreg_train import training
from nn import Brain
from sampler import sample
from variables import learning_rate, learning_rate_decay, sampling, scheduler_type, selected_features, steps, stochastic, labels_column

train_set = pd.read_csv("datasets/dataset_train.csv")
# Load the dataset
dataset_train = pd.read_csv("datasets/dataset_train.csv")
selectable_features = sorted(list(select_numerical_features(dataset_train).columns))
dataset_test = pd.read_csv("datasets/dataset_test.csv")


# Function to filter dataset based on selected features
def filter_dataset(selected_features):
    if selected_features:
        filtered_data = dataset_train[selected_features]
        return filtered_data
    else:
        return "No features selected"


def toggle_panel(stochastic):
    return gr.update(visible=stochastic)

def resample(rate):
    global dataset_test
    global dataset_train
    dataset_test, dataset_train = sample(train_set, rate)


def gradio_train(selected_features, learning_rate, steps, stochastic, learning_rate_decay, scheduler, sampling):
    if len(selected_features) == 0:
        return "No features selected"
    else:
        resample(sampling)
        scheduler = scheduler

        train_selected = get_selected_features(dataset_train, selected_features)
        x_train, quartiles = create_training_data(train_selected)

        features = x_train.columns.tolist()
        features_tensor = x_train.to_numpy()

        classes = create_classes(dataset_train)
        labels = create_labels(dataset_train, classes)
        labels_tensor = labels.to_numpy().T

        brain = Brain(classes, features)

        test_selected = get_selected_features(dataset_test, selected_features)
        x_test, _ = create_training_data(test_selected)

        losses, weights, accuracies = training(brain, features_tensor, labels_tensor, learning_rate, steps, stochastic, x_test, dataset_test)

        losses_df = pd.DataFrame(losses, columns=classes)
        weights_df = pd.DataFrame(weights, index=classes, columns=features)
        accuracy_df = pd.DataFrame(accuracies)

        figure = draw_graphs(losses_df, classes, accuracy_df)
        figure.suptitle(f"learning rate: {learning_rate} | Stochastic: {stochastic}")
        final_accuracy = f"{accuracies[-1] * 100}%"
        return figure, final_accuracy, weights_df, quartiles


def gradio_predict(selected_features, weights, quartiles):
    if weights is None or quartiles is None:
        return pd.DataFrame()
    if selected_features is None:
        return "No features selected"

    dataset = get_selected_features(dataset_test, selected_features)

    scaleddataset = robust_scale(dataset, quartiles).to_numpy()

    classes = weights.index
    features = weights.columns.tolist()

    brain = Brain(classes, features, weights=weights.to_numpy())

    prediction = predict(brain, scaleddataset)

    prediction["Truth"] = dataset_test[labels_column].values
    prediction["Good prediction"] = prediction["Truth"] == prediction[labels_column]
    return prediction


with gr.Blocks() as demo:
    weights_state = gr.State(None)
    quartiles_state = gr.State(None)
    with gr.Tab("Train"):
        with gr.Row():
            with gr.Column():
                sampling = gr.Slider(0, 0.90, step=0.01, value=sampling, label="Sampling rate", info="The percentage of the train set to sample")

                selected_features = gr.CheckboxGroup(choices=selectable_features, label="Select Features", value=selected_features)

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
            inputs=[selected_features, learning_rate, steps, stochastic, learning_rate_decay, scheduler, sampling],
            outputs=[figure, accuracy, weights_state, quartiles_state],
        )
    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column():
                prediction = gr.DataFrame(label="Prediction")
                predict_button = gr.Button(value="Predict")

            predict_button.click(fn=gradio_predict, inputs=[selected_features, weights_state, quartiles_state], outputs=[prediction])


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=False)
