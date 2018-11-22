import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('/home/hunaif/code/hunaif/titanic/data/train.csv')
holdout = pd.read_csv('/home/hunaif/code/hunaif/titanic/data/test.csv')
# explore(train)

def process_age(data,cut_points,label_names):
    data['Age'] = data['Age'].fillna(-0.5)
    data['Age_categories'] = pd.cut(data['Age'],bins=cut_points,labels=label_names)
    return  data

def convert_to_one_hot(data,column_name):
    dummy_col = pd.get_dummies(data[column_name],prefix=column_name)
    data = pd.concat([data,dummy_col],axis =1)
    return  data

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ['Missing','Infant','Child','Teenager','Young_Adult','Adult','Senior']

train = process_age(train,cut_points,label_names)
holdout = process_age(holdout, cut_points, label_names)

train = convert_to_one_hot(train,'Age_categories')
holdout = convert_to_one_hot(holdout, 'Age_categories')

train = convert_to_one_hot(train,'Pclass')
holdout = convert_to_one_hot(holdout, 'Pclass')

train = convert_to_one_hot(train,'Sex')
holdout = convert_to_one_hot(holdout, 'Sex')

print(train.head())
print(train.columns.values)


from sklearn.model_selection import train_test_split

columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
           'Age_categories_Missing','Age_categories_Infant',
           'Age_categories_Child', 'Age_categories_Teenager',
           'Age_categories_Young_Adult', 'Age_categories_Adult',
           'Age_categories_Senior']

all_X = train[columns]
all_y = train['Survived']
train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression()
#
# lr.fit(train_X, train_y)
# predictions = lr.predict(test_X)
#
# from sklearn.metrics import accuracy_score
# accuracy = accuracy_score(test_y, predictions)
# print(accuracy)
#
# from sklearn.metrics import confusion_matrix
# conf_matrix = confusion_matrix(test_y, predictions)
# conf_matrix = pd.DataFrame(conf_matrix,columns=[['Survived','Died']],index=[['Survived','Died']])
# print(conf_matrix)

from sklearn.model_selection import cross_val_score
import numpy as np

lr = LogisticRegression()
scores = cross_val_score(lr, all_X, all_y, cv=10)
print(np.mean(scores))


