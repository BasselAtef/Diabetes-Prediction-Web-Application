
from flask import Flask, render_template, request, url_for
import pickle

app = Flask(__name__)


clf = pickle.load(open('model.pkl', 'rb'))


app.static_folder = 'static'


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/diabetes')
def knowMore():
    return render_template('diabetes.html')


@app.route('/test', methods=["GET", "POST"])
def test():



    if request.method == "POST":

        myDict = request.form
        Preg = int(myDict['Preg'])
        GLU = float(myDict['GLU'])
        BP = float(myDict['BP'])
        ST = float(myDict['ST'])
        INS = float(myDict['INS'])
        BMI = float(myDict['BMI'])
        DPF = float(myDict['DPF'])
        Age = int(myDict['Age'])

        inputFeatures = (Preg, GLU, BP, ST, INS, BMI, DPF, Age)
        infProb = clf.predict([inputFeatures])[0]



        return render_template('show.html', inf=infProb)
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)
