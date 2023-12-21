import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
from IPython.display import display
from sklearn.tree import plot_tree

File_Path = 'D:/data/'
File_Name = 'car_data.csv'

df = pd.read_csv(File_Path + File_Name)

df.drop(columns=['User ID'],inplace= True)


encoders = []
for i in range(0, len(df.columns) - 1):
    enc = LabelEncoder()
    df.iloc[:, i] = enc.fit_transform(df.iloc[:, i])
    encoders.append(enc)
    
x = df.iloc[:, 0:3]
y = df['Purchased']
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)
display(x_train)
display(x_test)
display(y_train)
display(y_test)
print('random_state=0')
    
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x, y)


feature = x.columns.tolist()
Data_class= y.tolist()

plt.figure(figsize=(25, 20))
_ = plot_tree(model,
              feature_names= feature,
              class_names= Data_class,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=16)
plt.show()

import seaborn as sns
Feature_imp = model.feature_importances_
feature_names = ['Gender','Age','AnnualSalary'] 
 
sns.set(rc = {'figure.figsize' : (11.7,8.7)})
sns.barplot(x = Feature_imp, y = feature_names)
