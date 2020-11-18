import numpy as np
from python_speech_features import mfcc
from keras import Sequential
from keras.layers import Dense
#from sklearn.metrics import confusion_matrix
import scipy.io.wavfile as wav
if __name__=='__main__':
    k=0
    dataset_x = np.array([])
    dataset_y = np.array([])
    for k in ["03","04","09","49"]:
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
    classifier.fit(x_train,y_train, batch_size=20000, epochs=40)
    #end for
    eval_model=classifier.evaluate(x_train, y_train)
    eval_model
    (rate2,sig2) = wav.read("FS_P01_dev_035.wav")
    mfcc_feat2 = mfcc(sig2,rate2)
    x_test=mfcc_feat2
    y_pred=classifier.predict(x_test)
    y_pred =(y_pred>0.5)
    outp = []
    i=0
    for i in range(0,len(y_pred)):
        outp.append(int(y_pred[i][0]))
    f.close()
    f=open("FS_35_output.txt","w+")
    i=0
    for i in outp:
        f.write("{} ".format(i))
    f.close()
    
    f=open("FS_35_time_stamps.txt","w+")
    i=0
    j=0
    et=0
    try: 
        while(j<len(outp)):
            j=et
            if(outp[j]==0):
                st=j
                while(outp[j] != 1):
                    j=j+1
                et=j
                tag="NS"
            elif(outp[j]==1):
                st=j
                while(outp[j] != 0):
                    j=j+1
                et=j
                tag="S"
            f.write("{a}\t{b}\t{c}\n".format(a=st/100,b=et/100,c=tag))
    except(IndexError): 
        pass
        
        
        
        