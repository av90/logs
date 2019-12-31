# Revision

###  01. Difference between cross-validate and cross-val-score?
* `cross-val-score` is the most basic function for implementing cross validations
* In `cross-validate` specify multiple scoring methods
```python
# Applying K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=reg_lr, X= data['YearsExperience'].to_frame(), y=data['Salary'],cv=10,scoring='neg_mean_squared_error')
```

> All the values that in accuracies will be `negative`. This is because of the sklearn design selection which always select the maximum value to select the best performing model in both `reward` (increasing accuracy) and `penalty` (reducing error) scoring

### 02. Why we need to do LabelEncoding?
Because Machine Learning Model cannot handle non-numerical data so we need to make all text data to numerical
### 03. Why do we need to do OneHotEncoding?
After LabelEncoding, Text labels are converted to integers and when these integers are used to fit our ML model, it sees it as a `Ordinal` data which is not the case. So we need to do OneHotEncoding.

### 04. Why we need to omit one of the Dummy variable?
Because it is usually taken care by the constant part of the model. Just like the intercept in Linear model. 
If we don't omit, we falls into dummy variable trap which is our model is not able to distuingush the effect of one dummy variable as compare to other. 
>So whenever you create a dummy variable, you have to omit one

### 05. What is Linear/Multiple Regressions?
The aim of the model is to form a line called trend line that is able to predict all the observations with minimal error(`mean square error`)

### 06. What are the assumptions of Linear Regressions?
1. Linear relationship
2. Homoscaedasticity
3. Multi-variate Normality
4. Independence of the error
5. Lack of Multi-collinearity





