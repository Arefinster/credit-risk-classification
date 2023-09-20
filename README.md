# Credit-risk classification

## Analysis overview

This was an analytical exercise on supervised machine learning (ML) models to determine the creditworthiness of borrowers. The dataset used in this analysis included historical
record of lending activities from a peer-to-peer lending service company. The main purpose of the work  was to build, train and test various supervised ML models to predict the
status of a borrower as either a "healthy loan" or a "high-risk loan".

## Results

* The csv dataset was first imported into the Pandas dataframe. Then the "loan status" column was assigned as y and the rest were assigned as X. To obtained the unique record, 
y.value_counts() was used. The dataset was then split into train and test sets using the train_test_split module of sklearn.
* As model 1, LogisticRegression was instantiated from the sklearn.linear_model module. The classification report shows us the following information:

		precision    recall  f1-score   support
  healthy loan       1.00      1.00      1.00     18759
high-risk loan       0.87      0.89      0.88       625

      accuracy                           0.99     19384
     macro avg       0.94      0.94      0.94     19384
  weighted avg       0.99      0.99      0.99     19384

From the report, we see that precision, recall and f1-score for healthy loans using the LogisticRegression model are all 100%, however, for the high-risk loans, these accuracies
range from 87% to 89%.

* As model 2, RandomOverSampler is used to up sample the training data. And using this resampled data, the same LogisticRegression model is instantiated. The following report is
now generated.

		precision    recall  f1-score   support
  healthy loan       1.00      1.00      1.00     18759
high-risk loan       0.87      1.00      0.93       625

      accuracy                           1.00     19384
     macro avg       0.94      1.00      0.96     19384
  weighted avg       1.00      1.00      1.00     19384

Here we see that the precision remained the same, recall improved to 100% and f1-score improved to 93%. 

## Summary

We see that Model 2 shows better accuracies as it is able to detect the high-risk loans better than the previous model while also detecting the healthy loans with 100% accuracy. 
Therefore, we can recommend Model 2. I think it is also important to detect the high-risk loans with good accuracy in order to avoid financial loss.
