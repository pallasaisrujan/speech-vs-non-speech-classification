import numpy as np
from python_speech_features import mfcc
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
if __name__=='__main__':
    k=0
    dataset_x = np.array([])
    dataset_y = np.array([])
    for k in ["03","09","49"]:
        filename = "11-033" + k + ".txt"
        f=open(filename,"r")
        f1=f.readlines()
        ar = []
        i=0
        for x in f1:
            ar.append(x.split())
            i=0
            j=0
        t = []
        for i in range(0,len(ar)):
            for j in range(int(float(ar[i][0])*100),int(float(ar[i][1])*100)):
                if(j==0):
                    continue
                if(ar[i][2]=="NS"):
                    t.append(int(0))
                else:
                    t.append(int(1))
        (rate,sig) = wav.read("11-033" + k + "-1.wav")
        mfcc_feat = mfcc(sig,rate)
        if(k=="03"):
            dataset_x=mfcc_feat
            dataset_y=t
        else:
            dataset_x=np.concatenate((dataset_x,mfcc_feat),0)
            dataset_y=np.concatenate((dataset_y,t),0)
    x_train=dataset_x
    y_train=dataset_y
    classifier = Sequential()
    #First Hidden Layer
    classifier.add(Dense(6, activation='tanh', kernel_initializer='random_normal', input_dim=13))
    #Second  Hidden Layer
    classifier.add(Dense(7, activation='tanh', kernel_initializer='random_normal'))
    #Output Layer
    classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
    history = classifier.fit(x_train,y_train, batch_size=20000, epochs=40)
    #end for
    eval_model=classifier.evaluate(x_train, y_train)
    eval_model
    filename2 = "11-03304.txt"
    f22=open(filename2,"r")
    f2=f22.readlines()
    ar2 = []
    i=0
    for x2 in f2:
        ar2.append(x2.split())
    i=0
    j=0
    t2 = []
    for i in range(0,len(ar2)):
        for j in range(int(float(ar2[i][0])*100),int(float(ar2[i][1])*100)):
            if(j==0):
                continue
            if(ar2[i][2]=="NS"):
                t2.append(int(0))
            else:
                t2.append(int(1))
    (rate2,sig2) = wav.read("11-03304-1.wav")
    mfcc_feat2 = mfcc(sig2,rate2)
    x_test=mfcc_feat2
    y_test=t2
    y_pred=classifier.predict(x_test)
    y_pred =(y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    acc = cm[0][0]+cm[1][1]
    tot = cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
    acc = acc/tot
    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:")
    print(acc)
    
    
#print(history.history)
loss_train = history.history['loss']
accuracy = history.history['acc']
epochs = range(1,41)
plt.plot(epochs, loss_train, 'g', label='loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.title('Accuracy vs Epochs')
plt.plot(epochs, accuracy, 'b', label='accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
        