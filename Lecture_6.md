# Lecture 6
## Regularization in Logistic Regression
- Just like in linear regression, *regularization* can be performed for logistic regression
  - $\argmin _ \beta [ - \sum (y_i \ln(p_i) + (1-y_i)\ln(1 - p_i)) + \lambda \sum \beta_j^2]$
  - This will shrink the parameter estimates towards zero
- Choosing $\lambda$ can be done through strategies such as *k-fold* validation
- By regularizing, a logistic regression model could be made much more powerful by introducing more complexity without overfitting as much
  - For instance, a *polynomial* model could be utilized to allow for nonlinear decision boundaries
## Model Evaluation
- There are two types of *error* in binary classification
  - **False Positive**: Incorrectly predicting $\hat{Y} = 1$ when in truth $Y = 0$
  - **False Negative**: Incorrectly predicting $\hat{Y} = 0$ when in truth $Y = 1$
- The results of a classification algorithm can be summarized either via a **confusion matrix** or a **receiver operating characteristics (ROC) curve**
### Confusion Matrix
- ![Confusion Matrix](./Images/Confusion_Matrix.png)
  - An entry $CM_{i, j}$ indicates the number of tuples in class *i* that were labeled by the classifier as class *j*
- The *threshold* of a classification model can be changed - that is, classification could be based on $\hat{P}(Y = 1) > c$
  - The value of $c$ could be changed accordingly based off of the confusion table
  - Decreasing $c$ allows for more predictions of positives (class 1), which increases the number of *true positives* but also the number of *false positives*
  - Increasing $c$ allows for more predictions of negatives (class 0), which increases the number of *true negatives* but also the number of *false negatives*
  - Which threshold to choose should be based on the context of the data - based on whether false positives or false negatives are more severe 
    - e.g. One would want to avoid false negatives in the context of disease diagnosis 
- **Classifier Accuracy**: $\frac{TP + TN}{ALL}$
- **Error Rate**: $1 - accuracy = \frac{FP + FN}{ALL}$
- **Sensitivity**: True positive recognition rate = $\frac{TP}{P}$
- **Specificity**: True negative recognition rate = $\frac{TN}{N}$
- **Precision**: The percentage of tuples that the classifier labeled as positive that are actually positive
  - $\frac{TP}{TP + FP}$
- **Recall**: The percentage of positive tuples that the classifier labeled as positive
  - $\frac{TP}{TP + FN}$
- There is an inverse relationship between precision and recall
- **F-Measure**: $\frac{2 \times precision \times recall}{precision + recall}$
- **Weighted F Measure**: $\frac{(1 + \beta)^2 \times precision \times recall}{\beta_2 \times precision + recall}$