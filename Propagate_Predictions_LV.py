'''
Propagate LSTM Predictions
'''
import numpy as np
#Keras imports
from tensorflow.keras.models import load_model
from LSTM_LV import Lotka_Volterra
import matplotlib.pyplot as plt

#Load model
checkpoint_filepath='best_LV_LSTM.h5'
model = load_model('best_LV_LSTM.h5')

#Load point to be propagated
LV=Lotka_Volterra()
data=LV.data

initial_point=data['test'][0][0]
initial_point=initial_point.reshape((1, LV.sequence_length, LV.n_features))

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

res=np.array(predictions).reshape(steps,2)

#Plot Results
plt.plot(res[:250,0],label='Predator Population')
plt.plot(res[:250,1],label='Pray Population')
plt.title('Propagated Predictions')
plt.ylabel('Population')
plt.xlabel('Time step')
plt.legend(loc=1)
plt.savefig('./Figures/LV_Propagated_Predictions.pdf')
plt.show()

    