import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
mbanglore = pickle.load(open('banglore_home_prices_model.pkl', 'rb'))
mchennai = pickle.load(open('Chennai_home_prices_model.pkl', 'rb'))
mkolkata = pickle.load(open('Kolkata_home_prices_model.pkl', 'rb'))
mdelhi = pickle.load(open('Delhi_home_prices_model.pkl', 'rb'))
mmumbai = pickle.load(open('Chennai_home_prices_model.pkl', 'rb'))
mhyderabad = pickle.load(open('Hyderabad_home_prices_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('cities.html')

@app.route('/bengaluru')
def bengaluru():
    return render_template('index.html' , city_name = 'Bengaluru')

@app.route('/hyderabad')
def hyderabad():
    return render_template('index.html' , city_name = 'Hyderabad')

@app.route('/kolkata')
def kolkata():
    return render_template('index.html' , city_name = 'Kolkata')

@app.route('/chennai')
def chennai():
    return render_template('index.html' , city_name = 'Chennai')

@app.route('/mumbai')
def mumbai():
    return render_template('index.html' , city_name = 'Mumbai')

@app.route('/delhi')
def delhi():
    return render_template('index.html' , city_name = 'Delhi')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    cityname = features[0]
    int_features = features[1:]
    int_features = [int(x) for x in int_features]
    print(int_features)   
    final_features = [np.array(int_features)]
    if(cityname == "Bengaluru"):
        prediction = mbanglore.predict(final_features)
        output = round(prediction[0], 2)
        numbers = "{:,}".format(output)
        return render_template('index.html', prediction_text='Expected House Price is INR {}'.format(numbers) , city_name =cityname)
    if(cityname == "Hyderabad"):
        prediction = mhyderabad.predict(final_features)
        output = round(prediction[0], 2)
        numbers = "{:,}".format(output)
        return render_template('index.html', prediction_text='Expected House Price is INR {}'.format(numbers) , city_name =cityname)
    if(cityname == "Kolkata"):
        prediction = mkolkata.predict(final_features)
        output = round(prediction[0], 2)
        numbers = "{:,}".format(output)
        return render_template('index.html', prediction_text='Expected House Price is INR {}'.format(numbers) , city_name =cityname)
    if(cityname == "Chennai"):
        model = pickle.load(open('Chennai_home_prices_model.pkl','rb'))
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        numbers = "{:,}".format(output)
        return render_template('index.html', prediction_text='Expected House Price is INR {}'.format(numbers) , city_name =cityname)
    
    if(cityname == "Mumbai"):
        prediction = mmumbai.predict(final_features)
        output = round(prediction[0], 2)
        numbers = "{:,}".format(output)
        return render_template('index.html', prediction_text='Expected House Price is INR {}'.format(numbers) , city_name =cityname)
    

    if(cityname == "Delhi"):
        prediction = mdelhi.predict(final_features)
        output = round(prediction[0], 2)
        numbers = "{:,}".format(output)
        return render_template('index.html', prediction_text='Expected House Price is INR {}'.format(numbers) , city_name =cityname)
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)