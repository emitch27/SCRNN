import numpy as np
import pickle
import torch.nn as nn
import torch
import time
from datetime import datetime
import data, Laplacian, plotting, SCNN, train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########  Parameters  ###########
win_size = 0.1   #window size used for time binning
data_interval = 2
end_time = 10    #minutes to include for total data
# end_time = 'end'   #if set to 'end', total data includes all data
test_end_time = 2   #minutes of total data are used for testing; the rest are used for training


sequence_length = 5   #length of input sequence used for RNN component


epochs = 1
batch_size = 16
learning_rate = 0.001
dropout = 0.3

nn_layers = 3   #number of recurrent layers
hidden_size = 200   #dimension of hidden RNN components
rnn_activation = 'relu'   #activation function used in recurrent layers



now = str(datetime.now())
dt = str(now[:10]+ '_' + str(now[11:])) #for tracking time of operations
print('start time :', dt)


#########  Data  ###########
print('Converting to training data...')
#Load data
if end_time=='end': #use all data
	file = open('../analyses/' + str(win_size*1000) + 'ms/interval' + str(data_interval) + 'mod1.p', 'rb')
	spike_data_dict = pickle.load(file)
	file.close()
	spike_count_matrix1 = spike_data_dict['count_matrix']  #load spike count matrix of first module
	# x = spike_data_dict['x']
	# y = spike_data_dict['y']
	x = spike_data_dict['x'] + 0.75 #load ground truth x location (adding 0.75 puts location in [0,1.5] instead of [-0.75,0.75])
	y = spike_data_dict['y'] + 0.75 #load ground truth y location (adding 0.75 puts location in [0,1.5] instead of [-0.75,0.75])

	file = open('../analyses/' + str(win_size*1000) + 'ms/interval' + str(data_interval) + 'mod2.p', 'rb')
	spike_data_dict = pickle.load(file)
	file.close()
	spike_count_matrix2 = spike_data_dict['count_matrix']  #load spike count matrix of second module

	file = open('../analyses/' + str(win_size*1000) + 'ms/interval' + str(data_interval) + 'mod3.p', 'rb')
	spike_data_dict = pickle.load(file)
	file.close()
	spike_count_matrix3 = spike_data_dict['count_matrix']  #load spike count matrix of third module

	
else: #use data up until end time specified above
	stop_idx = int(end_time*60 / win_size)
	file = open('../analyses/' + str(win_size*1000) + 'ms/interval' + str(data_interval) + 'mod1.p', 'rb')
	spike_data_dict = pickle.load(file)
	file.close()
	spike_count_matrix1 = spike_data_dict['count_matrix'][:,:stop_idx]
	# x = spike_data_dict['x'][:stop_idx]
	# y = spike_data_dict['y'][:stop_idx]
	x = spike_data_dict['x'][:stop_idx] + 0.75
	y = spike_data_dict['y'][:stop_idx] + 0.75

	file = open('../analyses/' + str(win_size*1000) + 'ms/interval' + str(data_interval) + 'mod2.p', 'rb')
	spike_data_dict = pickle.load(file)
	file.close()
	spike_count_matrix2 = spike_data_dict['count_matrix'][:,:stop_idx]

	file = open('../analyses/' + str(win_size*1000) + 'ms/interval' + str(data_interval) + 'mod3.p', 'rb')
	spike_data_dict = pickle.load(file)
	file.close()
	spike_count_matrix3 = spike_data_dict['count_matrix'][:,:stop_idx]

spike_count_matrix = np.vstack((spike_count_matrix1, spike_count_matrix2, spike_count_matrix3))  #stack spike count matrices from all grid cell modules to create a single spike count matrix
in_size = spike_count_matrix.shape[0]   #define size of RNN inputs

test_stop_idx = int(test_end_time*60 / win_size)

#prepare data for loading for training
training_data = data.DatasetRNN(device, spike_count_matrix[:,test_stop_idx:], x[test_stop_idx:], y[test_stop_idx:], sequence_length)



#########  Network  ###########
network = SCNN.RNN(device, in_size, 2, hidden_size, nn_layers, rnn_activation, dropout).to(device)   #define network to be RNN
optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate) #Adam optimization
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.00) #learning rate scheduler
criterion = nn.MSELoss()   #loss function

data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)


print('Training network...')
train.train_RNN(network, device, data_loader, optimizer, criterion, scheduler, epochs) #train network



print('Plotting network prediction...')
plotting.plot_model_predictRNN(network, sequence_length, spike_count_matrix, x, y, test_stop_idx)   #plot results




