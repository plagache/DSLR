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



@app.route('/')
def homepage():
    return render_template('homepage.html', name='Homepage')


@app.route('/describe', methods=['POST', 'GET'])
def describe():
    table = False
    gryffindor = False
    hufflepuff = False
    ravenclaw = False
    slytherin = False

    if request.method == "POST":
        subprocess.call(['make webdescribe'], shell=True)
        # get html table

    if os.path.isfile('templates/describe_table.html'):
        table = True
    if os.path.isfile('templates/describe_table_gryffindor.html'):
        gryffindor = True
    if os.path.isfile('templates/describe_table_hufflepuff.html'):
        hufflepuff = True
    if os.path.isfile('templates/describe_table_ravenclaw.html'):
        ravenclaw = True
    if os.path.isfile('templates/describe_table_slytherin.html'):
        slytherin = True

    return render_template('describe.html', table=table, gryffindor=gryffindor, hufflepuff=hufflepuff, ravenclaw=ravenclaw, slytherin=slytherin)

@app.route('/histogram', methods=['POST', 'GET'])
def histogram():
    image_path = os.path.join('static', 'Image', 'hist')

    if request.method == "POST":
        subprocess.call(['make webhistogram'], shell=True)

    image_names = os.listdir(image_path)
    image_names.sort()
    images = [os.path.join(image_path, i) for i in image_names]

    return render_template('histogram.html', images=images)

@app.route('/scatter', methods=['POST', 'GET'])
def scatter():
    image_path = os.path.join('static', 'Image', 'scatter')

    if request.method == "POST":
        subprocess.call(['make webscatter'], shell=True)

    image_names = os.listdir(image_path)
    image_names.sort()
    images = [os.path.join(image_path, i) for i in image_names]

    return render_template('scatter.html', images=images)

@app.route('/pair', methods=['POST', 'GET'])
def pair():
    image_path = os.path.join('static', 'Image', 'pair')

    if request.method == "POST":
        subprocess.call(['make webpair'], shell=True)

    image_names = os.listdir(image_path)
    image_names.sort()
    images = [os.path.join(image_path, i) for i in image_names]

    return render_template('pair.html', images=images)
