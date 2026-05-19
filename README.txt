================================================================================
                    THYROID PROJECT - README
================================================================================

PROJECT DESCRIPTION
-------------------
This project is a web application that predicts thyroid disease outcomes using 
machine learning. The application is built using Flask and implements a 
pre-trained XGBoost model for making predictions based on medical data.


PROJECT STRUCTURE
-----------------
Thyroid project/
├── app.py                 - Main Flask application file
├── requirements.txt       - Python dependencies
├── README.txt            - This file

│
├── model/                - Directory containing pre-trained models
│   ├── thyroid_xgb_recur.pkl       - Trained XGBoost model
│   ├── label_encoders_recur.pkl    - Label encoders for categorical data
│   
│
├── static/               - Static web assets
│   ├── css/
│      ├── style.css     - Application styling
│       
│   
│
└── templates/            - HTML templates
    ├── index.html        - Main prediction interface
    


REQUIREMENTS
------------
The required Python packages are listed in requirements.txt:
- Flask==2.0.3
- pandas==1.3.5
- scikit-learn==1.0.2
- numpy==1.21.4
- xgboost==1.5.0


INSTALLATION
------------
1. Ensure Python 3.7+ is installed on your system
2. Navigate to the project directory
3. Install dependencies:
   pip install -r requirements.txt


RUNNING THE APPLICATION
-----------------------
Execute the following command from the project root directory:
   python app.py

The application will start and be available at:
   http://127.0.0.1:5000/


FEATURES
--------
- Web-based user interface for thyroid disease prediction
- Integration with pre-trained XGBoost machine learning model
- Input validation and preprocessing
- Real-time prediction results
- Responsive design with CSS styling


USAGE
-----
1. Open your web browser and navigate to http://127.0.0.1:5000/
2. Enter the required medical parameters
3. Click the prediction button
4. View the prediction results


NOTES
-----
- The model files (thyroid_xgb_recur.pkl and label_encoders_recur.pkl) 
  must be present in the model/ directory for the application to function
- Ensure all dependencies from requirements.txt are installed before running
- The application runs locally by default on port 5000



- **URL:** `/login`
- **Method:** `GET`, `POST`
- **Description:** Handles user login. Renders the dashboard on successful login.

### Dashboard

- **URL:** `/dashboard`
- **Method:** `GET`
- **Description:** Renders the dashboard page.

### Predict

- **URL:** `/predict`
- **Method:** `POST`, `GET`
- **Description:** Handles form submission for accident severity prediction and renders the result.

