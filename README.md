## 📍 Try it on Hugging Face
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/karthsanth/ai_grade_assesser)

**# Problem Statement**
- Develop an AI-based system to predict a student’s final grade using features like attendance, participation, assignment scores, and exam marks.
- Leverage machine learning models to provide accurate predictions, enabling educational institutions to better understand student performance and intervene when necessary.

**# Approach**

**1. Data Collection & Preprocessing**
- **Dataset**: Used the Student Performance Dataset, which contains features like school, sex, address, family size, parental education, and prior grades.
- **Data Cleaning**: Handled missing values and label-encoded categorical features to convert them into numerical representations.
- **Feature Selection**: Selected all columns as features except the final grade (G3), which was set as the target variable.

**2. Model Development**
- **Decision Tree Regressor**:
  - Trained a Decision Tree Regressor using an 80-20 train-test split.
  - Evaluated the model using Root Mean Square Error (RMSE) to assess prediction accuracy.
- **Artificial Neural Network (ANN)**:
  - Standardized data using a Standard Scaler to ensure better ANN performance.
  - **ANN Architecture**:
    - **Input Layer**: Number of neurons equal to the number of features.
    - **Hidden Layers**: Two hidden layers with 64 and 32 neurons respectively, both using ReLU activation.
    - **Output Layer**: Single neuron with linear activation for regression.
  - Compiled the model with the Adam optimizer and mean squared error loss function.
  - Trained the model for 50 epochs with a batch size of 32.

**3. Deployment**
- **Gradio**: Created an interactive web interface using Gradio to allow users to input student details and view the predicted grade from both the Decision Tree and ANN models.
- **Streamlit**: Created an alternative deployment using Streamlit, where users can enter student details and view grade predictions in real-time.

**# Results**

**1. Model Performance**
- **Decision Tree Regressor**:
  - **RMSE**: {Insert RMSE value}
  - **Feature Importance**: Key contributors to student performance included prior grades, study time, and absences.
  - **Feature Importance Visualization**: Visualized the relative importance of each feature in the Decision Tree model.
- **Artificial Neural Network (ANN)**:
  - **RMSE**: {Insert RMSE value}
  - **Loss Curve**: Training and validation loss curves showed model convergence over 50 epochs.
  - **Confusion Matrix (if classification method used)**: Displayed classification performance, if applicable.

**2. Visualizations**
- **Feature Importance Plot**: Displays the relative importance of each feature in the Decision Tree model.
- **Training Loss Plot**: Shows how the ANN’s training and validation loss changed over time.

**# Challenges**

**1. Data Quality Issues**
- **Challenge**: Categorical variables needed to be converted into numeric format.
- **Solution**: Used Label Encoding to convert categories into numerical values for training.

**2. Model Selection**
- **Challenge**: Choosing between regression and classification approaches.
- **Solution**: Tested both regression and classification models. Used RMSE for regression and confusion matrix/accuracy for classification.

**3. Deployment**
- **Challenge**: Deploying the models for user interaction.
- **Solution**: Used Gradio and Streamlit to create interactive web-based applications where users can input student details to predict final grades.


**# Future Improvements**
- Use a larger dataset to improve model generalization.
- Implement hyperparameter tuning for both models to enhance performance.
- Include additional visualizations to provide better insights into student performance trends.
- Enable export of predictions to CSV or PDF for institutional reporting.


