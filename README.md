# Parkinson's Disease Detection using XGBoost Classifier

This is a Python code for detecting Parkinson's Disease using XGBoost Classifier. It uses this [dataset](https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection).

## Attribute Information:

- **MDVP:Fo(Hz)**: Average vocal fundamental frequency
- **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency
- **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency
- **MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP**: Several measures of variation in fundamental frequency
- **MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA**: Several measures of variation in amplitude
- **NHR, HNR**: Two measures of ratio of noise to tonal components in the voice
- **status**: Health status of the subject (one) - Parkinson's, (zero) - healthy control


## Requirements

The following Python libraries are required to run the code:

- pandas
- seaborn
- matplotlib
- sklearn
- xgboost

You can install these libraries using pip:

```
pip install pandas seaborn matplotlib sklearn xgboost
```


## How it works

The code first reads the dataset from a CSV file and performs some data cleaning by dropping the 'name' column and checking for null values. Then, it uses a correlation matrix to visualize the correlations between the features.

Next, the dataset is split into training and testing sets using the `train_test_split()` function from the sklearn library. An XGBoost Classifier model is then created and fit to the training data.

The accuracy of the model is calculated using the `accuracy_score()` function from the sklearn library, and a confusion matrix is generated using the `confusion_matrix()` function to visualize the performance of the model.

- Accuracy of the model was: 0.9318 



## Credits
- This code was created by [Nada Osama](https://github.com/NadaOsamaa)
- [Parkinson's Disease Dataset](https://www.kaggle.com/datasets/debasisdotcom/parkinson-disease-detection) was obtained from Kaggle
