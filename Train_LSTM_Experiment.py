from azureml.core import Run
import matplotlib.pyplot as plt
from LSTM_LV import Lotka_Volterra
import os
# Get the experiment run context
run = Run.get_context()

#Define model and training parameters
n_features=2
n_steps=12
hidden_lstm=6
Dense_output=2
epochs=250
    
#Lotka Volterra data parameters
shift=1
sequence_length=12
sample_size=2000
    
#Build, train and evaluate the model
LV=Lotka_Volterra(shift,sequence_length,sample_size,hidden_lstm,Dense_output,n_steps,n_features,epochs)
model,history,score=LV.build_train_lstm()
print(f'Test loss: {score}')
#Log the score
run.log('Test Loss: ',score)
os.makedirs('./outputs', exist_ok=True)

# Visualize training history
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.plot(history.history['loss'],label='Training Loss')
plt.title('Validation loss history')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc=1)
plt.savefig('./outputs/Training_History.jpg')

run.complete()