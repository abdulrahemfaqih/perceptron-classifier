from flask import Flask, request, render_template, redirect, url_for
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open("model/perceptron_pickle", "rb") as r:
    perp = pickle.load(r)

LABEL = ["Paru", "Ekstra Paru"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = int(request.form['gender'])
    ft = int(request.form['ft'])
    hiv = int(request.form['hiv'])
    diabet = int(request.form['diabet'])
    tcm = float(request.form['tcm'])

    newdata = [[gender, ft, hiv, diabet, tcm]]
    result = perp.predict(newdata)
    result = LABEL[result[0]]
    
    return render_template('result.html', 
                           gender=gender, ft=ft, hiv=hiv, diabet=diabet, 
                           tcm=tcm, result=result)

if __name__ == "__main__":
    app.run(debug=True)