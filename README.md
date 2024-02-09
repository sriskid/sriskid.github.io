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
**Tags: Deep Learning, Reinforcement Learning, PyTorch, Deep-Q Learning**
- Created a custom environment to simulate the backjack card game that would return the state and reward values
- Created a deep learning model that would take the game state as input and output an action ("Hit", "Stand")
- Experimented with exploration vs. exploitation to allow the model to maximize rewards while also finding new strategies through random actions
- Plotted the results in real-time to show the models score over the course of thousands of games

### Car Damage Severity Classification
**Tags: Deep Learning, Computer Vision, PyTorch, Multiclass Image Classification, Data Visualization, Transfer Learning**
- Utilized EfficientNetB7 to classify damage severity of car damages (minor, moderate, severe)
- Fine tuned the pretrained model to suit the image classification
- Utilized Data Augmentation to increase the number of images and address overfitting
- Achieved 0.75 validation accuracy

### Intel Image Classification
**Tags: Deep Learning, Computer Vision, Tensorflow, Multiclass Image Classification, OpenCV, Data Visialization, Transfer Learning**
- Utilized VGG16 pretrained model and fine tuned it to classify different objects in drone images (buildings, forest, glacier, mountain, sea, street)
- Utilized an Image Data Generator to conserve memory and perform data augmentation to prevent overfitting
- Utilized Early Stopping and a Learning Rate Scheduler to prevent overfitting
- Achieved 0.93 accuracy on validation dataset
- Evaluated the model by plotting the loss, accuracy, and confusion matrix
- Tested the model on sample images

### Vehicle Damage Classification
**Tags: Deep Learning, Computer Vision, Data Visualization, OpenCV, PyTorch, Multiclass Image Classification, Transfer Learning**
- Utilized VGG16 pretrained model and fine tuned it to classify different types of car damage from images (crack, scratch, flat tire, dent, shattered glass, broken lamp)
- Utilized Early Stopping and a Learning Rate Scheduler to prevent overfitting
- Achieved 0.95 accuracy on validation dataset
- Evaluated the model by plotting the loss, accuracy, and confusion matrix
- Made predictions on sample images

### Skin Cancer Segmentation
**Tags: Deep Learning, Computer Vision, PyTorch, Binary Image Segmentation, Albumentations, Data Visualization**
- Created a model to segment skin cancers from images and create a mask to show the area of the cancer
- Utilized a U-Net model from scratch to take an RGB image as input and output a black and white mask (cancer in the white)
- Utilized Data Augmentation on the images and corresponding masks using albumentations to prevent overfitting
- Utilized Early Stopping and a Learning Rate Scheduler to prevent overfitting
- Achieved 0.96 pixel accuracy on the validation dataset
- Evaluated the model by loss and pixel accuracy
- Plotted predicted masks on the test images and compared to the ground truth

### Road Segmentation 
**Tags: Deep Learning, Computer Vision, PyTorch, Multiclass Image Segmentation, Transfer Learning**
- Utilized a pretrained U-Net model and fine tuned it to segment an image of a road into the background, road signs, cars, markings, and road surface
- Created a dictionary to map the RGB pixel values into different class for training and convert the classes of each pixel into RGB values for plotting
- Utilized DiceLoss for the loss function, a Learning Rate Scheduler to get better results, and Early Stopping to prevent overfitting
- Evaluated the model on loss and Intersection over Union (IOU)
- Achieved 0.97 IOU on the validation dataset
- Plotted predictions on the sample images with the color mapping and compared results to ground truth

### Ecommerce Text Classification
**Tags: Machine Learning, Deep Learning, Natural Language Processing, Scikit-Learn, Tensorflow**
- Created three models to classify descriptions into four labels (Household, Books, Electronics, and Clothing & Accessories)
- Vectorized the descriptions using a Bag-of-Words approach
- Utilized two classical machine learning models (Naive Bayes and SVM) to classify the descriptions and compared speed and accuracy
- For a deep learning approach, the tensorflow tokenizer was used along with padding to process the text, followed by a custom model to output the label
- The final results were that the Naive Bayes model took the least amount of time for 0.94 accuracy on the validation data and the Deep Learning model took the most with an acccuracy of 0.98 on the validation dataset

### Twitter Sentiment Analysis
**Tags: Machine Learning, Deep Learning, Natural Language Processing, Scikit-Learn, Tensorflow**
- Created six models to classify tweets by sentiment (Irrelevant, Negative, Neutral, Positive)
- Vectorized the descriptions using a Bag-of-Words approach for the ML models
- Utilized four classical ML models (Naive Bayes, SVM, Decision Tree, Random Forest, and XGBoost) and a Deep Learning model to compare speed/accuracy
- For a deep learning approach, the tensorflow tokenizer was used along with padding to process the text, followed by a custom model to output the label, and evaluated the model by loss and accuracy
- A final comparison showed that the Naive Bayes model was the fastest and had 0.83 validation accuracy, the SVM model took the longest with 0.94 validation accuracy, the XGBoost model performed the worst with 0.76 validation accuracy, and the deep learning model performed the best with 0.99 validation accuracy and took the third least amount of time

### Simple Steam Content-Based Recommendation System
**Tags: Data Science, Recommendation System, Natural Language Processing, Scikit-Learn, Exploratory Data Analysis**
- Created a simple content-based recommendation system to recommend games from the Steam game library
- Performed Exploratory Data Analysis to view trends in prices, reviews, operating systems, and ratings
- Vectorized the descriptions using CountVectorizer and calculating cosine similarities among the vectorized descriptions
- Output returns the games with the top 5 cosine similarities to the description of one's favorite game

### LEGO Minifigurines Generative Adversarial Network (GAN)
**Tags: Deep Learning, Generative AI, GANs, PyTorch**
- Created a GAN to generate images of minifigurines
- Utilized a generator to generate images (256 x 256 x 3) from a latent vector of dimension 128 and a discriminator to determine whether generated images are fake or not
- Trained for 3000 epochs and balanced the loss values of both adversarial models
- Plotted generated images from random latent vectors
