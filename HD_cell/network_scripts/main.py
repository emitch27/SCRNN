import numpy as np
import torch
import torch.nn as nn
import time
import pickle
import data, Laplacian, plotting, SCNN, train


#########  set device to gpu or cpu  ###########
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


#########  Parameters  ###########
session = 'Mouse28-140313'
win_size = 0.1   #window size used for time binning
end_time = 20   #minutes to include for total data
# end_time = 'end'   #if set to 'end', total data includes all data
test_end_time = 10   #minutes of total data are used for testing; the rest are used for training

intervals_per_sample = 3   #intervals included in individua inputs to SCNN
sequence_length = 5   #length of input sequence used for RNN component

threshold = 30   #threshold parameter used in binarization step of pre-processing
max_simplex_dim = 2  #max dimension of simplices included in functional simplicial complex
max_active = 8   #max number of active cells in a time bin

epochs = 2   
batch_size = 32
learning_rate = 0.001
dropout = 0.3

max_conv_dim = 1   #maximum dimension of convolutional filter used
sc_layers = 1   #number of simplicial convolutional layers
n_filters = 3   #number of filters per simplicial convolutional layer
degree = 2   #degree of simplicial filters
rnn_layers = 1   #number of recurrent layers
hidden_size = 50   #dimension of hidden components
conv_activation = 'relu'   #activation function used in simplicial convolutional layers
rnn_activation = 'relu'   #activation function used in recurrent layers


RNN=True   #flag for using RNN on backend; if False, fully connected layers are used instead




#########  Data  ###########
print('Converting to training data...')

#########  Load data  ###########
if end_time=='end':
	data_file = open('../analyses/' + '%0.0fms_win_size/'%(win_size*1000) + 'count_angle_time_' + '%s.p'%session, 'rb')
	spike_data_dict = pickle.load(data_file)
	data_file.close()
	spike_count_matrix = spike_data_dict['count_matrix']   #load spike count matrix
	angles = (spike_data_dict['angles_vec'] + np.pi) % (2*np.pi)#array
	times = spike_data_dict['times']
	spike_count_matrix, angles, times = data.takeout_nan(spike_count_matrix, angles, times) #remove columns of spike count matrix and elements of angles, times that contains nan values
else:
	stop_idx = int(end_time*60 / win_size)
	data_file = open('../analyses/' + '%0.0fms_win_size/'%(win_size*1000) + 'count_angle_time_' + '%s.p'%session, 'rb')
	spike_data_dict = pickle.load(data_file)
	data_file.close()
	spike_count_matrix = spike_data_dict['count_matrix']
	angles = (spike_data_dict['angles_vec'] + np.pi) % (2*np.pi)#array
	times = spike_data_dict['times']
	spike_count_matrix, angles, times = data.takeout_nan(spike_count_matrix, angles, times) #remove columns of spike count matrix that contains nan values
	spike_count_matrix, angles, times = spike_count_matrix[:,:stop_idx], angles[:stop_idx], times[:stop_idx]



###Binarize spike count matrix
binary_spike_count_matrix = data.binary_data_by_row(spike_count_matrix, threshold, max_active)   #generates binary spike count matrix

n_neurons, n_samples = spike_count_matrix.shape



print('Calculating Laplacians...')
#Calculate Laplacians for each piece of data
st_ind_dict = data.build_ind_dict(binary_spike_count_matrix, max_simplex_dim)
full_binary_st = Laplacian.build_all_complexes(binary_spike_count_matrix, st_ind_dict, max_simplex_dim)#List of lists of dictionaries representing simplicial complexes
bdry = Laplacian.build_boundaries(st_ind_dict)
Ups, Downs = Laplacian.build_ups_downs(bdry)
Up = [U.to_dense().clone().detach().to(device) for U in Ups]
Down = [D.to_dense().clone().detach().to(device) for D in Downs]

Up = data.Laplacian_power(Up, degree)#Returns list of lists of upper Laplacians raised to different powers
Down = data.Laplacian_power(Down, degree, ident=False)#Returns list of lists of lower Laplacians raised to different powers

Lap = []
Lap.append(Up[0])
for k in range(1, len(Up)):
	Lap.append(torch.cat([Up[k], Down[k-1]], dim=0))
Lap.append(Down[-1])




n_edges = len(st_ind_dict[1])
n_triangles = len(st_ind_dict[2])
n_simp_list = [n_neurons, n_edges, n_triangles]
print('   Neurons, edges, triangles :', n_simp_list)


test_stop_idx = int(test_end_time*60 / (win_size))


print('Calculating cochains...')
cochains = data.build_cochains(full_binary_st, spike_count_matrix, n_edges, n_triangles) #List of cochain tensors for each dimension. 


train_cochains = [C[...,test_stop_idx:] for C in cochains]



#prepare data for loading for training
if RNN:
	training_data = data.DatasetSCRNN(device, train_cochains, angles[test_stop_idx:], sequence_length)
elif intervals_per_sample==1:
	training_data = data.Dataset(device, train_cochains, angles[test_stop_idx:])
else:
	training_data = data.Dataset_gen(device, train_cochains, angles[test_stop_idx:], intervals_per_sample)



#########  Network  ###########
print('Building network...')
input_size = sum(n_simp_list[:(max_conv_dim+1)])

print('Flattened feature vector size:', input_size)



# load neural network 
if RNN: #SCRNN
	network = SCNN.SCNN_RNN(max_conv_dim, sc_layers, n_filters, sequence_length, n_simp_list, degree, Lap, input_size, rnn_layers, hidden_size, 1, dropout, \
	conv_activation, rnn_activation).to(device)
elif intervals_per_sample==1: #SCNN with only one time bin considered for each input
	network = SCNN.SCNN(max_conv_dim, sc_layers, n_filters, n_simp_list, degree, Lap, input_size, rnn_layers, nn_width, 1, dropout, \
	conv_activation, rnn_activation).to(device)
else: #SCNN with intervals_per_sample time bins considered for each input
	SCNN.SCNN(max_conv_dim, sc_layers, n_filters, intervals_per_sample, n_simp_list, degree, Lap, input_size, rnn_layers, nn_width, 1, dropout, \
	conv_activation, rnn_activation).to(device)


optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate) #Adam optimization
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998) #learning rate scheduler
criterion = nn.MSELoss()   #loss function

data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

# print('number of model parameters:', sum(p.numel() for _, p in network.named_parameters() if p.requires_grad))

print('Training network...')
train.train(network, device, data_loader, optimizer, criterion, scheduler, epochs) #train network



print('Plotting network prediction...')
#plot results
if RNN:
	plotting.plot_model_predictSCRNN(network, sequence_length, cochains, angles, times, test_stop_idx)
elif intervals_per_sample==1:
	plotting.plot_model_predict(network, cochains, angles, times, test_stop_idx)
else:
	plotting.plot_model_predict_gen(network, intervals_per_sample, cochains, angles, times, test_stop_idx)









