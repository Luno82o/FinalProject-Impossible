import pandas as pd
import matplotlib.pyplot as plt
import sys

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

sys.path.append("..")
from lib.action_setting import Actions



#------------------------------------------------------------------------
#Actions label code
#exchange to action_setting to define model classifer 
    
#------------------------------------------------------------------------
#load data.csv
def getData():   
    raw_data = pd.read_csv('data.csv', header=0)
    nrow=raw_data.shape[0] #row count
    ncol=raw_data.shape[1] #col count
    dataset = raw_data.values #all value ndarray
    
    value = dataset[0:nrow, 0:ncol-1].astype(float) #value ndarray
    #label = dataset[0:nrow, ncol-1] #label ndarray
    
    nclass_dict=raw_data['label'].value_counts() #label dictionary
    
    label_encoder=[] #create label code list
    for x in range(len(nclass_dict)):
        label_encoder=label_encoder+([x]*nclass_dict[Actions(x).name])
    
    ###check label_encoder is correct count
    #from collections import Counter
    #tmp=Counter(label_encoder)
    #print(nclass_dict,tmp)
    
    label_onehot = np_utils.to_categorical(label_encoder) #one-hot encoding
    
    return value,label_onehot

#------------------------------------------------------------------------
#training result plot
class LossHistory(Callback):
    
    def __init__(self):
        self.task_type=''
    
    #if train start, it will trigger event
    def on_train_begin(self, logs={}):
        self.task_type='train'
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
        print("Training is over.")
    
    #if every batch end, it will trigger event
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        
    #if execution cycle(epoch) start, it will trigger event
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch=epoch
        print(f"--------The strat {self.task_type} of the No.{epoch+1}---------- .")
        
    #if execution cycle(epoch) end, it will trigger event
    def on_epoch_end(self, epoch, logs={}):
        print(f"----------The end {self.task_type} of the No.{epoch+1}---------- .")
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
    
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

    
#------------------------------------------------------------------------
#train/test Features

#train 80%,test 20% and set randrom seed
train_Features,test_Features,train_Label,test_Label = train_test_split(getData()[0],getData()[1],test_size=0.2,random_state=777)
print('--------------------------------------------')
print('total:',len(getData()[0]),', train:',len(train_Features),', test:',len(test_Features))
print('--------------------------------------------')

#------------------------------------------------------------------------
# (models Sequential)---->create models network
model = Sequential()
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=4, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=len(Actions), activation='softmax')) #output layer


#------------------------------------------------------------------------
# (models compile)---->create training solution method
            #define loss fun, optimizer fun, mertrics(衡量指標)
model.compile(loss='categorical_crossentropy',optimizer=Adam(0.0001), metrics=['accuracy']) #0.0001學習率


#------------------------------------------------------------------------
# (models fit)---->execute training process
lh=LossHistory()
model.fit(train_Features,train_Label,
          epochs=35,    #執行X次訓練週期
          batch_size=32, #每一個週期執行Y筆資料
          verbose=2,    #顯示訓練過程
          validation_data=(test_Features, test_Label),   #加入驗證集讓資料跑得更好
          callbacks=[lh]  #在訓練的過程中回call其他函數
          )

model.summary()
lh.loss_plot('epoch')
model.save('models.h5')

#------------------------------------------------------------------------
# (models predict)---->predict accuracy
loss, accuracy = model.evaluate(test_Features,test_Label,batch_size=32)
print('Test loss:{:.3}'.format(loss))
print('Test accuracy:{:.3}'.format(accuracy))



#------------------------------------------------------------------------
#confusion matrix
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


Y_pred = model.predict(test_Features)
cfm = confusion_matrix(np.argmax(test_Label,axis=1), np.argmax(Y_pred, axis=1))
np.set_printoptions(precision=2)

plt.figure()
class_names = ['stand', 'squat']
plot_confusion_matrix(cfm, classes=class_names, title='Confusion Matrix')
plt.show()

