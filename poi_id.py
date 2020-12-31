#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_payments','bonus','total_stock_value','long_term_incentive']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


#data exploration
print('there are {} data points'.format(len(data_dict)))

from itertools import islice
print(list(islice(data_dict.items(),2)))

number_of_poi = 0
for(n , record) in data_dict.items():
    if record['poi'] == 1:
        number_of_poi += 1
print('number of POI is : {}'.format(number_of_poi))

### Task 2: Remove outliers
max = 0
for item in data_dict.items():
    if isinstance(item[1]['salary'], int) and isinstance(item[1]['bonus'], int):
        salary = int(item[1]['salary'])
        bonus = int(item[1]['bonus'])
        if salary > max:
            max = salary
        matplotlib.pyplot.scatter( salary, bonus )
print(max)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()


data_dict.pop('TOTAL',0)
### Task 3: Create new feature(s)
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))

bonus = []
salary = []
total_stock_value = []
total_payments = []

for item in data_dict.items():
    if item[1]['bonus'] == 'NaN' or item[1]['salary'] == 'NaN' or item[1]['total_stock_value'] == 'NaN' or item[1]['total_payments'] == 'NaN':
        bonus.append(0.)
        salary.append(0.)
        total_stock_value.append(0.)
        total_payments.append(0.)
    else:
        bonus.append(item[1]['bonus'])
        salary.append(item[1]['salary'] )
        total_stock_value.append(item[1]['total_stock_value'])
        total_payments.append(item[1]['total_payments'])

bonus = np.array(bonus).reshape(-1, 1)
salary = np.array(salary).reshape(-1, 1)
total_stock_value = np.array(total_stock_value).reshape(-1, 1)
total_payments = np.array(total_payments).reshape(-1, 1)

        
bonus_scaled = min_max_scaler.fit_transform(bonus)
salary_scaled= min_max_scaler.fit_transform(salary)
stock_scaled = min_max_scaler.fit_transform(total_stock_value)
payments_scaled = min_max_scaler.fit_transform(total_payments)

j = 0
for item in data_dict.items():
    if bonus_scaled[j][0] == 0. or salary_scaled[j][0] == 0. or stock_scaled[j][0] == 0. or payments_scaled[j][0] == 0.:
        item[1]['bonus_salary_ratio'] = 0.
        item[1]['stock_payments_ratio'] = 0.
    else :
        item[1]['bonus_salary_ratio'] = bonus_scaled[j][0] /salary_scaled[j][0] 
        item[1]['stock_payments_ratio'] = stock_scaled[j][0] / payments_scaled[j][0]
    j+= 1

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score , precision_score , recall_score

#clf = GaussianNB()
#the accuracy score is :0.8604651162790697
#the recall score is :0.4
#the precision score is :0.4

#clf = tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
#the accuracy score is :0.8837209302325582
#the recall score is :0.4
#the precision score is :0.5

#clf = SVC()
#error

#clf = RandomForestClassifier( random_state=10)
#the accuracy score is :0.8837209302325582
#the recall score is :0.2
#the precision score is :0.5

#clf = LogisticRegression(class_weight= 'balanced')
#the accuracy score is :0.5348837209302325
#the recall score is :0.4
#the precision score is :0.10526315789473684

#clf = KNeighborsClassifier(n_neighbors=3)
#the accuracy score is :0.813953488372093
#the recall score is :0.0
#the precision score is :0.0

clf1 = AdaBoostClassifier(n_estimators=100, random_state=0)
estimators = [('reduce_dim', PCA(n_components=5)), ('clf', clf1)]
clf = Pipeline(estimators)
#the accuracy score is :0.8372093023255814
#the recall score is :0.4
#the precision score is :0.3333333333333333

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

accuracy = accuracy_score(labels_test,pred)
recall = recall_score(labels_test,pred)
precision = precision_score(labels_test,pred)

print('the accuracy score is :{}'.format(accuracy))
print('the recall score is :{}'.format(recall))
print('the precision score is :{}'.format(precision))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
