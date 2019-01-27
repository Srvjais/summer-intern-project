
# # Installing External Dependencies

get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')


# # Importing Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# # Reading CSV Files

data=pd.read_csv('Dataset2.csv')
data.drop('Sample code number',axis=1,inplace=True)
data.head()

# #  Features of Dataset

categories=["benign","malignant"]
data.describe()


# # Data Augmentation using Re-encoding and Data Imputation

data['2b,4m'].replace([2,4],[0,1],inplace=True)
mean=round(data.mean()[6])
data['Bare Nuclei']=data['Bare Nuclei'].replace('?',mean)


# # One hot encoding

values = data.values
X=values[:,:-1]
y=values[:,-1]
y = np_utils.to_categorical(y, len(categories))


# # Making data set using train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42,stratify=y)


# # Training the Model


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, input_dim=9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=100, batch_size=10,validation_data=(X_test,y_test))
scores = model.evaluate(X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# # Accuracy-95.71%

import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics


def plot_history(history):
    if history.get('acc') and history.get('val_acc'):
        plt.plot(history['acc'], marker='.', label='train_accuracy')
        plt.plot(history['val_acc'], marker='.', label='validation_accuracy')
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


# # Accuracy vs Epochs(knees at 10 epochs & saturates at 40 epochs)
# 
# # Loss vs Epochs
# 
# 

plot_history(history.history)


# # Predicted Y for X_test  (Solution of  Question 1)


y_pred=model.predict(X_test)


# # Save the model

def save_model(model):
    with open(os.path.join(".","saved") + '.json', 'w')as model_file:
        model_file.write(model.to_json())
    model.save_weights(os.path.join(".","saved") + '.h5')
    print('Saved model successfully')
save_model(model)


# # Optimize the Threshold to minimize false negative
# 
# threshold is a function which takes parameters as y_pred and a value(like 0.5) above which we define the tumor is malignant.
# 
# We vary the threshold for predicting the malignant tumor and plot the respective false negatives and positives to inspect the optimal threshold at which the false negative rate is appreciably low and false positive rate is not noticibly high.
# 
# From inspection we find such a point at threshold=0.6.This means that when our model is >60% sure that the tumor is malignant,we 
# are safe to inform the patient.
# 

def threshold(y_pred,val):
    y=y_pred[:,1]>val
    y=y.astype("uint8")
    y = np_utils.to_categorical(y, len(categories))
    return y

margins=np.arange(0.05,1,step=.05)
false_negatives=[]
false_positives=[]
for margin  in margins:
    y_pred_curr=threshold(y_pred,margin)
    confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred_curr, axis=1)).astype('float')
    ((true_positive,false_negative),(false_positive,true_negative))=confusion_matrix
    false_positives.append(false_positive)
    false_negatives.append(false_negative)


plt.plot(margins,false_negatives,label="false_negatives")
plt.plot(margins,false_positives,label="false_positives")
plt.legend()
plt.xlabel("thresholds")


y_pred_total=model.predict(X)

y_pred_curr=threshold(y_pred_total,0.6)
confusion_matrix = sklearn.metrics.confusion_matrix(np.argmax(y, axis=1), np.argmax(y_pred_curr, axis=1)).astype('float')
((true_positive,false_negative),(false_positive,true_negative))=confusion_matrix
print("False Positive",false_positive)
print("False Negative",false_negative)
print("number of test cases",len(y_pred_total))
print("on the threshold of p=",0.6)


# # Conclusion
# In the dataset of 699 cases we find only 2 cases where we fail to predict the existance of malignant tumor in the patient,
# We also have 17 patients who are erroneously informed of having a malignant tumor.
