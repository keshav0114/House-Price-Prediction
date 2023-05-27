import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
mbanglore = pickle.load(open('bangalore_home_prices_model.pkl', 'rb'))
mchennai = pickle.load(open('Chennai_home_prices_model.pkl', 'rb'))
mkolkata = pickle.load(open('Kolkata_home_prices_model.pkl', 'rb'))
mdelhi = pickle.load(open('Delhi_home_prices_model.pkl', 'rb'))
mmumbai = pickle.load(open('Chennai_home_prices_model.pkl', 'rb'))
mhyderabad = pickle.load(open('Hyderabad_home_prices_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/bengaluru')
def bengaluru():
    return render_template('cities.html' , city_name = 'Bengaluru')

@app.route('/hyderabad')
def hyderabad():
    return render_template('cities.html' , city_name = 'Hyderabad')

@app.route('/kolkata')
def kolkata():
    return render_template('cities.html' , city_name = 'Kolkata')

@app.route('/chennai')
def chennai():
    return render_template('cities.html' , city_name = 'Chennai')

@app.route('/mumbai')
def mumbai():
    return render_template('cities.html' , city_name = 'Mumbai')

@app.route('/delhi')
def delhi():
    return render_template('cities.html' , city_name = 'Delhi')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    #features = [x for x in request.form.values()]
    cityname = features[5]
    area = int(features[0])
    beds = int(features[1])
    powerbackup = features[2]
    gasconnection = features[3]
    airconditioner = features[4]

    # Map powerbackup, gasconnection, and airconditioner to binary values
    powerbackup = 1 if powerbackup.lower() == 'yes' else 0
    gasconnection = 1 if gasconnection.lower() == 'yes' else 0
    airconditioner = 1 if airconditioner.lower() == 'yes' else 0
  
     # Map the form values to the corresponding feature names in your CSV file
    feature_names = ['Area', 'No. of Bedrooms','Powerbackup', 'Gasconnection', 'AC']
    feature_values = [area, beds, powerbackup, gasconnection, airconditioner]

    # Create a dictionary to map the form values to the CSV feature names
    feature_dict = dict(zip(feature_names, feature_values))

    # Retrieve the feature values in the same order as in your CSV file
    final_features = [np.array([feature_dict['Area'],
                                feature_dict['No. of Bedrooms'],
                                feature_dict['Powerbackup'],
                                feature_dict['Gasconnection'],
                                feature_dict['AC']])]

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
    app.run(debug=True,port=4000)