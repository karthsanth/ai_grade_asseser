**Problem Statement**

The goal of this project is to develop an AI-based system that predicts a student's final grade
using relevant features like attendance, participation, assignment scores, and exam marks.
The objective is to leverage machine learning models to provide accurate predictions,
enabling educational institutions to better understand student performance and intervene
when necessary.

**Approach**

1. Data Collection & Preprocessing
● Dataset: The Student Performance Dataset was used, which contains features like
school, sex, address, family size, parental education, and prior grades.
● Data Cleaning: Missing values were handled, and categorical features were
label-encoded to convert them into numerical representations.
● Feature Selection: The feature set included all columns except the final grade (G3),
which served as the target variable.

2. Model Development
● Decision Tree Regressor:
○ A Decision Tree Regressor was trained using an 80-20 train-test split.
○ The model was evaluated using the Root Mean Square Error (RMSE) to
assess prediction accuracy.
● Artificial Neural Network (ANN):
○ The data was standardized using a Standard Scaler to ensure better
performance of the ANN.
○ The ANN architecture included:
■ Input Layer: Number of neurons equal to the number of features.
■ Hidden Layers: Two hidden layers with 64 and 32 neurons
respectively, both using ReLU activation.
■ Output Layer: A single neuron with linear activation for regression.
○ The model was compiled with the Adam optimizer and mean squared error
loss function.
○ The model was trained over 50 epochs with a batch size of 32.

3. Deployment
● Gradio: An interactive web interface was created using Gradio to allow users to input
student details and view the predicted grade from both the Decision Tree and ANN
models.
● Streamlit: An alternative deployment was created using Streamlit, where users can
enter student details and view grade predictions in real time.
Results

1. Model Performance
● Decision Tree Regressor:
○ RMSE: {Insert RMSE value}
○ Feature Importance: The key contributors to student performance included
prior grades, study time, and absences.
○ Feature Importance Visualization:
● Artificial Neural Network (ANN):
○ RMSE: {Insert RMSE value}
○ Loss Curve: Training and validation loss curves showed model convergence
over 50 epochs.
○ Confusion Matrix (if classification method used):
**2. Visualizations**
● Feature Importance Plot: Displays the relative importance of each feature in the
Decision Tree model.
● Training Loss Plot: Shows how the ANN's training and validation loss changed over
time.
**Challenges**
1. Data Quality Issues
● Challenge: Categorical variables needed to be converted into numeric format.
● Solution: Used Label Encoding to convert categories into numerical values for
training.
2. Model Selection
● Challenge: Choosing between regression and classification approaches.
● Solution: Both regression and classification models were tested. RMSE was used
for regression, and confusion matrix and accuracy were used for classification.
3. Deployment
● Challenge: Deploying the models for user interaction.
● Solution: Used Gradio and Streamlit to create interactive web-based applications
where users can input student details to predict final grades.
Usage Instructions
1. Run Locally:
○ Install required libraries:
pip install -r requirements.txt
○ Run the Streamlit application:
streamlit run app.py
○ Or, run the Gradio interface:
python app.py
2. Interact with the Web Application:
○ Enter student details (like age, address, parental education, and previous
grades) into the input fields.
○ View the predicted final grade from both the Decision Tree and ANN models.
3. Files:
○ app.py: Contains the Gradio and Streamlit application logic.
○ decision_tree_model.pkl: Saved Decision Tree model.
○ ann_model.h5: Saved ANN model.
○ README.md: Project documentation.
Future Improvements
● Use a larger dataset to improve model generalization.
● Implement hyperparameter tuning for both models to enhance performance.
● Include additional visualizations to provide better insights into student performance
trends.
● Enable export of predictions to CSV or PDF for institutional reporting
