
 ## Installing External Dependencies

get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install seaborn')


 ## Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Loading Dataset 

data=pd.read_csv('Dataset1.csv')
data.head(3)


# ## Combining the Hotel Name and Hotel Room Features

# Hotel room is not independent of the actual hotel name. Precisely, Room 1 in Hotel A is not the same as Room 1 in Hotel B. 
#Hence we need to concatenate these features to form a new feature and drop the old hotel rooms feature.

data['Hotel-Room']=data['Hotel name']+data['Nr. rooms'].map(str)
data.drop('Hotel name',axis=1,inplace=True)
data.drop('Nr. rooms',axis=1,inplace=True)
data.head(3)


data['Hotel stars']=data['Hotel stars'].apply(lambda x:np.mean(list(map(int,x.split(',')))))
data.dtypes


# ## Removing Outliers
# The column Member years contains an outlier having value as -ve. We remove this row completely to avoid polluting the dataset.

print(data.describe())
idx=(data['Member years']<0).idxmax()
data.drop(idx,inplace=True)


# ## One Hot Encoding the Categorical Classes
# For categorical quantities it is essential to one hot encode them (LabelBinarization). 
#We cannot simply label encode them as they do not form an intrinsically continuous series. 
#We here provide code for both these methods but OneHotEncodeing is always superior so we use that one.
# Moreover, if we require to process further data, we will need the encoders used on the training dataset so we store them in the encoders dictionary for later use.



from collections import defaultdict
from sklearn.preprocessing import LabelBinarizer
encoders = defaultdict(LabelBinarizer)
categorical_columns = data.select_dtypes(include=['object']).columns
for categorical_column in categorical_columns:
    enc=encoders[categorical_column]
    one_hot_encoding=enc.fit_transform(data[categorical_column])
    if len(enc.classes_) ==2:
        data[categorical_column+' binary'] = one_hot_encoding
    else:
        for i,column in enumerate(encoders[categorical_column].classes_):
            data[column]=one_hot_encoding[:,i]
data.drop(categorical_columns, axis=1, inplace=True)
data.head()

#Not Used
"""
    from collections import defaultdict
    from sklearn.preprocessing import LabelEncoder
    encoders = defaultdict(LabelEncoder)
    categorical_columns = data.select_dtypes(include='object').columns
    data[categorical_columns]=data[categorical_columns].apply(lambda x: encoders[x.name].fit_transform(x))
    data.head()
""";


# # Append the 'score' column to the end

cols=data.columns.tolist()
cols.remove('Score')
cols.append('Score')
data=data[cols]
print(cols)


# ## Convert to numpy ndarray and convert y to categorical variable
# It is the requrement of keras that multiclass data should be converted into categorical format.

from keras.utils import np_utils
categories=['score '+str(i) for i in range(5)]
X=data.drop('Score',axis=1).values
y=data['Score'].values-1

y = np_utils.to_categorical(y, len(categories))
print(X.shape)
print(y.shape)

from collections import Counter
Counter(list(np.argmax(y,axis=1)+1))


# ## Split Data into Train and Test sets
# For validation we need to keep a portion of the dataset seperate to evaluate our model later.
# `train_test_split` also automatically shuffles the dataset so validation is more accurate.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42,stratify=y)


# ## First Attempt: Random Forest
# We see that random forest is not capable of handling this dataset as it performs poorly even with generous depth.

from sklearn.ensemble  import RandomForestClassifier
dt = RandomForestClassifier(n_estimators=200, max_depth=18,random_state=0)
history=dt.fit(X_train, y_train)
dt.score(X_test, y_test)


# ## Attempt Two: Neural Net
# Performs significantly better and is able to double the performance 

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD,RMSprop,Adagrad
sgd=Adagrad(lr=0.02, epsilon=.1, decay=1e-5)
model = Sequential()
model.add(Dense(20, input_shape=X.shape[1:], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(categories),activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=50, batch_size=10,validation_data=(X_test,y_test),verbose=True)
scores = model.evaluate(X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


import os
import sklearn.metrics


def plot_history(history):
    if history.get('acc') and history.get('val_acc'):
        plt.plot(history['acc'], marker='.', label='train_accuracy')
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.plot(history['val_acc'], marker='.', label='validation_accuracy')
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.title('Model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()
        plt.figure()
    

    if history.get('loss') and history.get('val_loss'):
        plt.plot(history['loss'], marker='.', label='train_loss')
        plt.plot(history['val_loss'], marker='.', label='validation_loss')
        plt.title('Model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()


def save_metrics(y_test, y_pred, categories):
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    classification_report = sklearn.metrics.classification_report(y_test, y_pred, target_names=categories, digits=5)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    confusion_matrix = confusion_matrix.astype('float')
    

    plt.imshow(confusion_matrix, interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return confusion_matrix,classification_report


# # Accuracy vs Epochs and Loss vs Epochs

plot_history(history.history)

# ## Predicted Y for X_test  (Solution of  Question 1)

y_pred=model.predict(X_test)

# ## Saving the Model

def save_model(model):
    with open(os.path.join(".","saved") + '.json', 'w')as model_file:
        model_file.write(model.to_json())
    model.save_weights(os.path.join(".","saved") + '.h5')
    print('Saved model successfully')
save_model(model)

# # Confusion Matrix

y_pred=model.predict(X)
confusion_matrix,classification_report=save_metrics(y,y_pred,categories)
print(classification_report)


# # Identifying the most relevent features

# We have used 'Tree based feature selection' for identifying the most relevent features.
#Tree-based estimators can be used to compute feature importances, which in turn can be used to discard irrelevant feature

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
classi = ExtraTreesClassifier(n_estimators=50)
classi = classi.fit(X, y)
arr=classi.feature_importances_
indices=arr.argsort()[-5:][::-1]

for index in indices:
    print(data.columns[index])


# # Conclusion
 
# Through the dataset we found that '5' is the score ,given by maximum reviewers.And also after training the model we found that
# 1)Nr. reviews
# 2)Nr. hotel reviews
# 3)Helpful votes
# 4)Member years
# 5)Hotel stars
# are the most relevent features that determine the score given by a reviewer.





