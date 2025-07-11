import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import pickle

# Importing the required dataset
dataset = pd.read_csv('Titanic-Dataset.csv')

#Filling in the missing data
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

# Combining the Siblings/Spouses column and the Parents/Children column to form Family Size
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']

# Drop Unnecessary Columns
dataset.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Creating the feature matrix and dependent variable vector
X = dataset.drop(['Survived','Cabin'],axis=1).values
y = dataset['Survived'].values

# Encoding the Embarked and Sex column
le = LabelEncoder()
ct = ColumnTransformer(transformers=[('encoders',OneHotEncoder(),[-2])],remainder='passthrough')
X[:,1] = le.fit_transform(X[:,1])
X = ct.fit_transform(X)

# Splitting the train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Training the Random Forest model on the train sets
classifier = RandomForestClassifier(n_estimators=50,random_state=0)
classifier.fit(X,y)

#Measuring the accuracy of the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f'The accuracy score is : {accuracy:.4f}')

# Displaying the confusion matrix and it's heatmap
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)