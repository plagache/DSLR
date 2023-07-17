# import base64
# from io import BytesIO
#
# from flask import Flask
# from matplotlib.figure import Figure
# import matplotlib.pyplot as pyplot
#
# app = Flask(__name__)
#
#
# @app.route("/")
# def homepage():
#     # Generate the figure **without using pyplot**.
#     fig = Figure()
#     ax = fig.subplots()
#     ax.plot([1, 2])
#     # Save it to a temporary buffer.
#     buf = BytesIO()
#     fig.savefig(buf, format="png")
#     # Embed the result in the html output.
#     buf.seek(0)
#     buf = pyplot.imread("ressources/hist.png")
#     data = base64.b64encode(buf.getbuffer()).decode("ascii")
#     return f"<img src='data:image/png;base64,{data}'/>"

from flask import Flask, render_template, request
import os
import subprocess

app = Flask(__name__)

image_path = os.path.join('static', 'Image', 'hist')


@app.route('/')
def plot_url():
    return render_template('homepage.html', name='Homepage')


@app.route('/describe', methods=['POST', 'GET'])
def describe():
    table = False

    if request.method == "POST":
        subprocess.call(['make webdescribe'], shell=True)
        # get html table

    if os.path.isfile('templates/describe_table.html'):
        table = True

    return render_template('describe.html', table=table)

@app.route('/histogram', methods=['POST', 'GET'])
def plot_histogram():

    if request.method == "POST":
        subprocess.call(['make webhistogram'], shell=True)

    images = os.listdir(image_path)
    images = [os.path.join(image_path, i) for i in images]

    return render_template('histogram.html', images=images)
