from flask import Flask, render_template, request, redirect, url_for
# from flask import Flask, render_template, request, redirect, url_for
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# import pandas as pd
# import os
# import joblib

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = './uploads'  # Folder to save uploaded files

# Ensure the upload folder exists
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model
# model = tf.keras.models.load_model('./Models/rnn_stock_model.h5')
# scaler = joblib.load('./Models/task2_scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    # if request.method == 'POST':
    #     file = request.files['data_file']
    #     if file and file.filename.endswith('.csv'):
    #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #         file.save(filepath)
            
    #         # Read the CSV
    #         data = pd.read_csv(filepath)
            
    #         # Assuming the last 10 rows of the CSV are the data for the last 10 days
    #         last_10_days = data[:10]
    #         input_data = last_10_days.drop(columns=["Date"]).values  # Drop Date column, and adjust if you have other columns

    #         # Normalization
    #         # scaler = MinMaxScaler(feature_range=(0, 1))
    #         input_data_normalized = scaler.fit_transform(input_data)

    #         # Reshape for the model
    #         input_seq = np.array([input_data_normalized])
        
    #         # Predict the next closing price
    #         predicted_price_normalized = model.predict(input_seq)
    #         ps_list = [[0,0,0,predicted_price_normalized[0][0],0,0]]
    
    #         # Inverse scale the predicted price
    #         predicted_list = scaler.inverse_transform(ps_list)
    #         prediction = predicted_list[0][3]
    return render_template('task2.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
