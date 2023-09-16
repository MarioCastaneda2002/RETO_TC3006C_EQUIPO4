from flask import Flask, request, jsonify, render_template
from modelo import predecir 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        var1 = float(request.form['var1'])
        var2 = float(request.form['var2'])
        var3 = float(request.form['var3'])
        var4 = float(request.form['var4'])
        var5 = float(request.form['var5'])
        var6 = float(request.form['var6'])

        
        prediction = predecir(var1, var2, var3, var4, var5, var6)

        
        return render_template('PW.html', prediction=prediction)

    return render_template('PW.html')

if __name__ == '__main__':
    app.run(debug=True)

