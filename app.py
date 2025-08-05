import numpy as np
import pandas as pd
from flask import Flask,request,render_template
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData,PredictPipeline
print("Successfully imported.")

application = Flask(__name__) # Create Flask app
app=application

@app.route("/")
def index():
    return render_template('index.html')   # Load HTML form

@app.route("/predictdata", methods=["GET", "POST"]) # Route for form submission and prediction

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        data=CustomData(                             # Get input values from form
        Type = request.form.get('Type'),
        Air_temperature_K = float(request.form.get('Air_temperature_K')),
        Process_temperature_K = float(request.form.get('Process_temperature_K')),
        Rotational_speed_rpm = int(request.form.get('Rotational_speed_rpm')),
        Torque_Nm = float(request.form.get('Torque_Nm')),
        Tool_wear_min = int(request.form.get('Tool_wear_min')),
        Target = float(request.form.get('Target'))
            
            
        )
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0]) #Show result on the same page
    
    

if __name__=="__main__":               # Start the Flask server
    app.run(host="0.0.0.0", debug=True)