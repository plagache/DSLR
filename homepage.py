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

from flask import Flask, render_template
import os

app = Flask(__name__)

image_path = os.path.join('static', 'Image')


@app.route('/')
def plot_url():
    file = os.path.join(image_path, 'hist.png')
    return render_template('hist.html', name= 'Histogram' ,image=file)
