import gradio as gr
import pandas as pd

from data_preprocessing import split_dataframe
from logreg_predict import predict
from logreg_train import training


# Function to select only the features list
def load_features(data):
    return


# Load the dataset
dataset_train = pd.read_csv("datasets/dataset_train.csv")
dataset_test = pd.read_csv("datasets/dataset_test.csv")


# Function to filter dataset based on selected features
def filter_dataset(selected_features):
    if selected_features:
        filtered_data = dataset_train[selected_features]
        return filtered_data
    else:
        return "No features selected"


# def train()
#           return


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
        # code_editor = gr.Code(value=load_code(), language="python", label="Edit variables.py")

        # output_message = gr.Textbox(label="Output")
        # Button to save the code
        # save_button = gr.Button("Save changes")
        # save_button.click(fn=save_code, inputs=[code_editor], outputs=output_message)
        gr.Interface(
            fn=filter_dataset,
            inputs=gr.CheckboxGroup(choices=list(dataset_train.columns), label="Select Features"),
            outputs=gr.DataFrame(label="Filtered Dataset"),
        )

    # with gr.Tab("Predict"):
    #     gr.Interface(fn=predict, inputs="file", outputs=gr.DataFrame(label="houses.csv"))

    # with gr.Tab("split_dataset"):
    #     gr.Interface(
    #         fn=split_dataframe,
    #         inputs=["file", gr.Number(value=float)],
    #         outputs=["file", "file"],
    #     )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=False)
