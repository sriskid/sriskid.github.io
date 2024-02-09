# Welcome
#### Technical Skills: Python, SQL, R

## Education
- M.S. Computational Biology | Carnegie Mellon University (_May 2023_)
- B.S. Biology | The University of Texas at Dallas (_May 2020_)

## Projects
### Credit Card Fraud Detection 
**Tags: Data Science, Exploratory Data Analysis, Data Exploration, Feature Engineering, Preprocessing, Scikit-Learn**
- Created models to detect fraudulent credit card transactions based on 28 features as well as transaction amaounts
- Performed Exploratory Data Analysis (EDA) on the dataset to investigate distribution of features by transaction types and feature correlations
- Engineered features by scaling feature values, taking into account outliers
- Trained Logistic Regression, Decision Tree, Random Forest, and XGBoost Classifiers and compared performances and speeds
- Investigated feature reductions based on the top 5 important features from the Random Forest and PCA and compared performances

### Stock Price Prediction
**Tags: Data Science, Exploratory Data Analysis, Time Series Analysis, Correlation Analysis, Lag Analysis, Seasonal Analysis, Scikit-Learn**
- Performed Time Series Analysis on Stock Price Data to Predict Stock Closing Prices
- Performed Lag Analysis through autocorrelation plots
- Engineered features to include lag data up to five days
- Trained Random Forest and XGBoost Models to predict on the test data
- Extrapolated future stock closing prices

### Blackjack Reinforcement Learning Agent
**Tags: Reinforcement Learning, PyTorch, Deep-Q Learning**
- Created a custom environment to simulate the backjack card game that would return the state and reward values
- Created a deep learning model that would take the game state as input and output an action ("Hit", "Stand")
- Experimented with exploration vs. exploitation to allow the model to maximize rewards while also finding new strategies through random actions
- Plotted the results in real-time to show the models score over the course of thousands of games

### Car Damage Severity Classification
**Tags: Deep Learning, PyTorch, Multiclass Image Classification, Data Visualization, Transfer Learning**
- Utilized EfficientNetB7 to classify damage severity of car damages (minor, moderate, severe)
- Fine tuned the pretrained model to suit the image classification
- Utilized Data Augmentation to increase the number of images and address overfitting
- Achieved 0.75 validation accuracy

### Intel Image Classification
**Tags: Deep Learning, Tensorflow, Multiclass Image Classification, OpenCV, Data Visialization, Transfer Learning**
- Utilized VGG16 pretrained model and fine tuned it to classify different objects in drone images (buildings, forest, glacier, mountain, sea, street)
- Utilized an Image Data Generator to conserve memory and perform data augmentation to prevent overfitting
- Utilized Early Stopping and a Learning Rate Scheduler to prevent overfitting
- Achieved 0.93 accuracy on validation dataset
- Evaluated the model by plotting the loss, accuracy, and confusion matrix
- Tested the model on sample images

### Vehicle Damage Classification
**Tags: Deep Learning, Data Visualization, OpenCV, PyTorch, Multiclass Image Classification, Transfer Learning**
- Utilized VGG16 pretrained model and fine tuned it to classift different types of car damage from images (crack, scratch, flat tire, dent, shattered glass, broken lamp)
- Utilized Early Stopping and a Learning Rate Scheduler to prevent overfitting
- Achieved 0.95 accuracy on validation dataset
- Evaluated the model by plotting the loss, accuracy, and confusion matrix
- Made predictions on sample images
