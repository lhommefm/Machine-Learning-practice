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
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

# histograms by variable
dataset.hist()

# scatter plot matrix between variables to look at relationships
scatter_matrix(dataset)

## SKLEARN TO TRAIN AND TEST THE MODEL
# python slice the columns in the NumPy array: https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/
# Split-out validation dataset, with columns 0-3 as X and 4 as the Y value
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
# what is name,model in models?
# what is StratifiedKFold?
# what is cross_val_score?
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()