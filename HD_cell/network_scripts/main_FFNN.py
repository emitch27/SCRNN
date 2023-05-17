import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import data, Laplacian, plotting, SCNN, train



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

#########  Parameters  ###########
session = 'Mouse28-140313'   #which session of data to decode
win_size = 0.1   #window size used for time binning
end_time = 20    #minutes to include for total data
# end_time = 'end'   #if set to 'end', total data includes all data
test_end_time = 10   #minutes of total data are used for testing; the rest are used for training


epochs = 2
batch_size = 32
learning_rate = 0.001
dropout = 0.3

nn_layers = 3   #number of layers of NN
nn_width = [128, 128, 64]    #list of width of layers of neural network
mlp_activation = 'relu'   #activation function used in FFNN





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
	angles = (spike_data_dict['angles_vec'] + np.pi) % (2*np.pi)
	times = spike_data_dict['times']
	spike_count_matrix, angles, times = data.takeout_nan(spike_count_matrix, angles, times)
	spike_count_matrix, angles, times = spike_count_matrix[:,:stop_idx], angles[:stop_idx], times[:stop_idx]


#Take out nan values
in_size = spike_count_matrix.shape[0]

test_stop_idx = int(test_end_time*60 / win_size)

training_data = data.DatasetFFNN(spike_count_matrix[:,test_stop_idx:], angles[test_stop_idx:])



#########  Network  ###########
network = SCNN.MLP(in_size, nn_layers, nn_width, 1, dropout, mlp_activation).to(device)   #define network to be FFNN
optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate) #Adam optimization
# criterion = nn.MSELoss()   #loss function
criterion = nn.L1Loss()

data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

print('number of model parameters:', sum(p.numel() for _, p in network.named_parameters() if p.requires_grad))
print('Training network...')
train.trainFFNN(network, device, data_loader, optimizer, criterion, epochs) #train network


#plot results
print('Plotting network prediction...')
plotting.plot_model_predictFFNN(network, spike_count_matrix, angles, times, test_stop_idx)












