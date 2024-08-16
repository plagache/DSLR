import os
import subprocess

from flask import Flask, render_template, request, json, redirect, url_for
from variables import selected_features

app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template("homepage.html", name="Homepage")


@app.route("/describe", methods=["POST", "GET"])
def describe():
    table = False
    gryffindor = False
    hufflepuff = False
    ravenclaw = False
    slytherin = False

    if request.method == "POST":
        subprocess.call(["make webdescribe"], shell=True)
        # get html table

    if os.path.isfile("templates/describe_table.html"):
        table = True
    if os.path.isfile("templates/describe_table_Gryffindor.html"):
        gryffindor = True
    if os.path.isfile("templates/describe_table_Hufflepuff.html"):
        hufflepuff = True
    if os.path.isfile("templates/describe_table_Ravenclaw.html"):
        ravenclaw = True
    if os.path.isfile("templates/describe_table_Slytherin.html"):
        slytherin = True

    return render_template("describe.html", table=table, gryffindor=gryffindor, hufflepuff=hufflepuff, ravenclaw=ravenclaw, slytherin=slytherin)


@app.route("/histogram", methods=["POST", "GET"])
def histogram():
    image_path = os.path.join("static", "Image", "hist")

    if request.method == "POST":
        subprocess.call(["make webhistogram"], shell=True)

    image_names = os.listdir(image_path)
    image_names.sort()
    images = [os.path.join(image_path, i) for i in image_names]

    return render_template("histogram.html", images=images)


@app.route("/scatter", methods=["POST", "GET"])
def scatter():
    image_path = os.path.join("static", "Image", "scatter")

    if request.method == "POST":
        subprocess.call(["make webscatter"], shell=True)

    image_names = os.listdir(image_path)
    image_names.sort()
    images = [os.path.join(image_path, i) for i in image_names]

    return render_template("scatter.html", images=images)


@app.route("/pair", methods=["POST", "GET"])
def pair():
    image_path = os.path.join("static", "Image", "pair")

    if request.method == "POST":
        subprocess.call(["make webpair"], shell=True)

    image_names = os.listdir(image_path)
    image_names.sort()
    images = [os.path.join(image_path, i) for i in image_names]

    return render_template("pair.html", images=images, features=selected_features)


def parse_value(value):
    try:
        return float(value)
    except ValueError:
        if value.lower() in ["true", "1", "yes"]:
            return True
        elif value.lower() in ["false", "0", "no"]:
            return False
        else:
            raise ValueError(f"Unexpected value format: {value}")


@app.route("/train", methods=["POST", "GET"])
def train():
    json_path = os.path.join("static", "variables.json")

    with open(json_path, "r") as json_file:
        parsed_json = json.load(json_file)
        features = {key: value for key, value in parsed_json.items() if isinstance(value, bool)}

    if request.method == "POST":
        form_data = request.form.to_dict()
        form_data.pop("type")
        parsed_data = {}

        for key in parsed_json.keys():
            if key in form_data:
                parsed_data[key] = parse_value(form_data[key])
            else:
                parsed_data[key] = False if isinstance(parsed_json[key], bool) else parsed_json[key]

        # for key in parsed_json.keys():
        #     if key in form_data:
        #         parsed_data[key] = parse_value(form_data[key])
        #     else:
        #         parsed_data[key] = False if isinstance(parsed_json[key], bool) else parsed_json[key]

        with open(json_path, "w") as file:
            json.dump(parsed_data, file, indent=4)
        return redirect(url_for("train"))
        match request.form["type"]:
            case "kfold":
                print("kfold")
            case "update":
                print("update")
            case "train":
                print("train")
        # subprocess.call(["make train"], shell=True)

    return render_template("train.html", content=parsed_json, data=features)
