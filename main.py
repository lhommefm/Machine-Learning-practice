# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

## PANDAS TO LOAD AND REVIEW THE DATA
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# shape - returns (# instances, # attributes)
print(F'Shape: \n {dataset.shape}')

# head - show first # entries
print(F'Head: \n {dataset.head(20)}')

# descriptions - show basic stats (count, mean, std, min/max, percentiles)
print(F'Descriptions: \n {dataset.describe()}')

# class distribution determined by grouping by a column and getting the size
print(F"Class size: \n {dataset.groupby('class').size()}")

# box and whisker plots by variable
print('Boxplots')
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# histograms by variable
print('Historgrams')
dataset.hist()

# scatter plot matrix between variables to look at relationships
print('Scatter Matrix')
scatter_matrix(dataset)

## SKLEARN TO TRAIN AND TEST THE MODEL
# python slice the columns in the NumPy array: https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
# Split-out validation dataset, with columns 0-3 as X and 4 as the Y value
# train_test_split to automatically split out training and validation datasets if the dataset is large enough
array = dataset.values
x = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

# Set up Algorithms provided by SKLearn
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
# name, model was defined above as tuple entries in the list
results = []
names = []
for name, model in models:
	# StratifiedKFold generates test sets (folds) such that each contain the same distribution of classes
    # Typically 5-10 folds are used depending on data set size
    # Model is fit across all K-1 folds combinations and validated on the remaining fold; compute average score
    # StratifiedKFold(number of splits, shuffle class' samples before splitting, ordering of indices if shuffled)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    # cross_val_score returns the score of each test fold
    # cross_val_score(model used to fit the data, x array, y array, cross-validation strategy, scoring metric)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Make predictions on validation dataset using winning model
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Accuracy Score: fraction of correctly classified samples
print(F'Accuracy Score: {accuracy_score(Y_validation, predictions)}')

# Confusion matrix: number of observations known to be in group (row) vs. predicted to be in group (column)
print(F'Confusion Matrix: \n{confusion_matrix(Y_validation, predictions)}')

# Classification report: classification metrics by class and overall
# Recall: class correctly identified / total of that class
# Precision: how many correctly classfied among that class
# F1-Score: harmonic mean between precision and recall
# Support: number of occurances of the given class in the dataset 
print(F'Classification Report: \n{classification_report(Y_validation, predictions)}')

## REFERENCE
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/