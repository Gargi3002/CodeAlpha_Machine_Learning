# CodeAlpha_Machine_Learning
# Handwritten cahracter recognition
This is just the initial basic project built using the mnist dataset which is capable of reconizing various handwritten digits(0-9) and can be extended to reconize alphabets or even words.
> MNIST dataset  contains 70,000 grayscale images of handwritten digits.
> loaded using fetch_openml('mnist_784',version=1)
> Uses classification Model:Random Forest Classifier with 100 estimators
> Evaluation Metrics:Accuracy,Classification Report,Confusion matrix
# How to run?
1.Open handwriting_recognition.ipynb in Jupyter Notebook.
2. Run all the cells sequentially.
3. The model is trained on the MNIST dataset and saved as mnist_model.pkl.
4. Load the model to make predictions on sample images.
# Disease prediction System
This predicts the likelihood of disease based on medical data(symptoms,patient history).It also uses classification algorithms to predict outcomes.
> Heart Disease Dataset from kAggle.Preloaded from heart_disesse.csv
> Uses Random Forest classifier with 100 estimators
> Evaluation Metrics:Accuracy,Classification Report,Confusion Matrix
# How to run?
1.Open disease_prediction.ipynb in Jupyter Notebook.
2. Run all the cells sequentially.
3. The model is trained on the heart_disease.csv and saved as disease_prediction_model.pkl.
4. Load the model to make predict the likelihood pf disease for new patients.
# BEFORE RUNNING INSTALL THE LIBRARIES IN JUPYTER NOTREBOOK AS
pip install numpy pandas matplotlib seaborn scikit-learn joblib.
