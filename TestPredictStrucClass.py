import os
#os.chdir('/Users/hastingj/Work/Python/chemont/chemont-struc/experiments')

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
import pickle

from chebidblite.learnhelper import ChebiDataPreparer

#from chebidblite import setupdb

#setupdb.prepareCacheAndIndex()

# ChebiDataPreparer from dblite/learnhelper
#dprep = ChebiDataPreparer()
#dprep.buildDefaultDataMaps()
#chemdata500x100x1024 = dprep.getDataForLearning(class_size=100,number_to_select=500)
#chemdata500x25x1024 = dprep.getDataForLearning(class_size=25,number_to_select=500)

#chemdata = chemdata500x100x1024
#chemdata = chemdata500x25x1024

with open('data/chemdata500x100x1024.pkl','rb') as output:
    #pickle.dump(chemdata,output)
    chemdata = pickle.load(output)

exit()

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Split-out validation dataset
array = chemdata.values
X = array[:,3:1027]  # Fingerprint columns length 1024, todo: make this configurable
Y = array[:,0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# COMPARE FLAT, SHALLOW ALGORITHMS
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('LDA', LinearDiscriminantAnalysis()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Save results with parameter description to a file
with open("results/compare_alg_results_500x100x1024.csv",'w') as outfile:
    for name,result in zip(names,results):
        outfile.write(name+",")
        outfile.write(", ".join([str(r) for r in result]))
        outfile.write("\n")

# Display Figure of Algorithms Comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


classif_reports = {}
# Make predictions on validation datasets
for name, model in lr_models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(name, accuracy_score(Y_validation, predictions))
    #print(confusion_matrix(Y_validation, predictions))
    classifres = classification_report(Y_validation, predictions, output_dict=True)
    classif_reports[name] = classifres

    f1s = [classifres[k]['f1-score'] for k in classifres.keys() if "CHEBI" in k]

    # Density Plot and Histogram of F1 scores
    sns.distplot(f1s, hist=True, kde=True,
                bins=int(180/5), color = 'darkblue',
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth': 4})
    plt.title('Density plot of F1 scores')
    plt.ylabel('Scores')
    plt.show()


# On 100x100 task, LR consistently performs best, followed by LDA then KNN. NB is the worst.

with open('data/classif_reports_200x100x1024.pkl','wb') as output:
    pickle.dump(classif_reports,output)

# Keep a record of the best F1 scores (the LR model) with their ChEBI IDs

classifres = classif_reports['LR']
lr_200x100x1024_f1s = [classifres[k]['f1-score'] for k in classifres.keys() if "CHEBI" in k]
lr_200x100x1024_chebis = [k for k in classifres.keys() if "CHEBI" in k]


with open ("results/lr_f1_200x100x1024.csv",'w') as outfile:
    for chebi_id, f1score in zip(lr_200x100x1024_chebis,lr_200x100x1024_f1s):
        outfile.write(chebi_id+", "+str(f1score)+"\n")

subsetter = ChebiOntologySubsetter()
subsetter.createSubsetFor(lr_200x100x1024_chebis)
colour_numbers = {k:v for (k,v) in zip(lr_200x100x1024_chebis,lr_200x100x1024_f1s)}
subsetter.printImageOfSubset(image_name="results/network_lr_200x100x1024.png", colour_nums=colour_numbers)


