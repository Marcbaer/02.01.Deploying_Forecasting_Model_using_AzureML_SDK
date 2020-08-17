'''
Propagate LSTM Predictions
'''
import numpy as np
#Keras imports
from tensorflow.keras.models import load_model
from Train_LSTM_LV import load_data
import matplotlib.pyplot as plt
#Load model
checkpoint_filepath='best_LV_LSTM.h5'
model = load_model('best_LV_LSTM.h5')

#Load point to be propagated
shift=1
sequence_length=12
pred_mode=1
n_features=2

data=load_data(shift,pred_mode,sequence_length)

initial_point=data['test'][0][0]
initial_point=initial_point.reshape((1, sequence_length, n_features))

#propagate predictions into the future
steps=300
predictions=[]
history=[initial_point]
for i in range(steps):
    pred=model.predict(history[-1])
    predictions.append(pred)
    pred=pred.reshape(1,1,2)
    new_point=np.append(history[-1][:,1:,:],pred,axis=1)
    history.append(new_point)
    print('step {}/{} predicted'.format(i,steps))

res=predictions.reshape(steps,2)

#Plot Results
plt.plot(res[:250,0],label='Predator Population')
plt.plot(res[:250,1],label='Pray Population')
plt.title('Propagated Predictions')
plt.ylabel('Population')
plt.xlabel('Time step')
plt.legend(loc=1)
plt.savefig('./Figures/LV_Propagated_Predictions.pdf')
plt.show()

    