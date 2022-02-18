from flask import Flask, render_template, request
import pickle

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html", Predicted_flower_name="Predicted flower name")


@app.route('/', methods=['POST'])
def predictandoutput():
    sl = request.form["slengthin"]
    sw = request.form["swidthin"]
    pl = request.form["plengthin"]
    pw = request.form["pwidthin"]
    input = [[sl, sw, pl, pw]]
    loaded_model = pickle.load(open('model_knn', 'rb'))
    result = loaded_model.predict(input)
    return render_template("index.html", Predicted_flower_name=result[0])


if __name__ == '__main__':
    loaded_model = pickle.load(open('model_knn', 'rb'))
    app.run(debug=True)
