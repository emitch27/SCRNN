import gudhi as gd
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import pickle
import data, Laplacian, plotting, SCNN, train



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


#########  Parameters  ###########
session = 'Mouse28-140313'
win_size = 0.1   #window size used for time binning
end_time = 20    #minutes to include for total data
# end_time = 'end'   #if set to 'end', total data includes all data
test_end_time = 10   #minutes of total data are used for testing; the rest are used for training
sequence_length = 5   #length of input sequence used for RNN component


epochs = 2
batch_size = 32
learning_rate = 0.001
dropout = 0.3

nn_layers = 1   #number of recurrent layers
hidden_size = 50   #dimension of hidden RNN components
rnn_activation = 'relu'   #activation function used in recurrent layers





#########  Data  ###########
print('Converting to training data...')
#Load data
if end_time=='end':
	data_file = open('../analyses/' + '%0.0fms_win_size/'%(win_size*1000) + 'count_angle_time_' + '%s.p'%session, 'rb')
	spike_data_dict = pickle.load(data_file)
	data_file.close()
	spike_count_matrix = spike_data_dict['count_matrix']   #load spike count matrix
	angles = (spike_data_dict['angles_vec'] + np.pi) % (2*np.pi)   #load array (vector) of ground truth HD angles
	times = spike_data_dict['times']   #load array (vector of same length as angles) of time (in seconds) of recording of HD angles 
	spike_count_matrix, angles, times = data.takeout_nan(spike_count_matrix, angles, times) #remove columns of spike count matrix and elements of angles, times that contains nan values
else:
	stop_idx = int(end_time*60 / win_size)
	data_file = open('../analyses/' + '%0.0fms_win_size/'%(win_size*1000) + 'count_angle_time_' + '%s.p'%session, 'rb')
	spike_data_dict = pickle.load(data_file)
	data_file.close()
	spike_count_matrix = spike_data_dict['count_matrix']
	angles = (spike_data_dict['angles_vec'] + np.pi) % (2*np.pi)#array
	times = spike_data_dict['times']
	spike_count_matrix, angles, times = data.takeout_nan(spike_count_matrix, angles, times)
	spike_count_matrix, angles, times = spike_count_matrix[:,:stop_idx], angles[:stop_idx], times[:stop_idx]


#Take out nan values
in_size = spike_count_matrix.shape[0]

test_stop_idx = int(test_end_time*60 / win_size)

training_data = data.DatasetRNN(spike_count_matrix[:,test_stop_idx:], angles[test_stop_idx:], sequence_length)



#########  Network  ###########
network = SCNN.RNN(device, in_size, 1, hidden_size, nn_layers, rnn_activation, dropout)   #define network to be RNN
optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate) #Adam optimization
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998) #learning rate scheduler
# criterion = nn.MSELoss()   #loss function
criterion = nn.L1Loss()

data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

print('number of model parameters:', sum(p.numel() for _, p in network.named_parameters() if p.requires_grad))
print('Training network...')
train.trainRNN(network, device, data_loader, optimizer, criterion, scheduler, epochs)   #train network



print('Plotting network prediction...')
#plot results
plotting.plot_model_predictRNN(network, sequence_length, spike_count_matrix, angles, times, test_stop_idx)












