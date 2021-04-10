import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# from IPython.display import display

currentDirectory = os.getcwd()
final_train = pd.read_csv(currentDirectory+"/final_train.csv") 

train = pd.read_csv('./unbalanced/train.csv')
test = pd.read_csv('./unbalanced/test.csv')

cols_to_drop = ['void()', 'subject']

train = train.drop(cols_to_drop, axis = 1)
test = test.drop(cols_to_drop, axis = 1)

target = ['Activity']

train_x, train_y = train.drop(target, axis = 1), train[target]
test_x, test_y = test.drop(target, axis = 1), test[target]


######################
#Making of forest
######################

number_trees = [10,20,40,80,100,150,200,500]

names_atributes = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
result = pd.DataFrame(None, index = names_atributes)

def make_forest(number_trees, test_x, train_x, train_y, *args):
  for trees in number_trees:
    if args:
      clf_basic_gini = RandomForestClassifier(trees, class_weight = args[0])
    else: 
      clf_basic_gini = RandomForestClassifier(trees)
    clf_basic_gini.fit(train_x, train_y.values.flatten())
    pred_y_basic_gini= clf_basic_gini.predict(test_x)
    temp = list()
    for i in range(6):
      temp.append("{:.3f}".format((precision_score(test_y, pred_y_basic_gini,average=None))[i])+"/"+"{:.3f}".format((recall_score(test_y, pred_y_basic_gini,average=None))[i])+"/"+"{:.3f}".format((f1_score(test_y, pred_y_basic_gini,average=None))[i]))
    if args:
      result[("random forest with " + str(trees) + "trees with " + args[0] + " weight")] = temp
    else:
      result[("random forest with " + str(trees))+ "trees "] = temp

make_forest(number_trees, test_x, train_x, train_y, "balanced")

make_forest(number_trees, test_x, train_x, train_y)

print(result)