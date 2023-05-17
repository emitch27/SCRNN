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

intervals_per_sample = 3   #intervals included in individua inputs to SCNN
sequence_length = 5   #length of input sequence used for RNN component

threshold = 5   #threshold parameter used in binarization step of pre-processing
max_simplex_dim = 2  #max dimension of simplices included in functional simplicial complex
max_active = 8   #max number of active cells in a time bin

epochs = 1
batch_size = 16
learning_rate = 0.001
dropout = 0.3

max_conv_dim = 2   #maximum dimension of convolutional filter used
sc_layers = 2   #number of simplicial convolutional layers
n_filters = 5   #number of filters per simplicial convolutional layer
degree = 2   #degree of simplicial filters
rnn_layers = 3   #number of recurrent layers
hidden_size = 200   #dimension of hidden components
conv_activation = 'relu'   #activation function used in simplicial convolutional layers
mlp_activation = 'relu'   #activation function used in recurrent layers
1

RNN=True   #flag for using RNN on backend; if False, fully connected layers are used instead


#time stamp for book keeping
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

#Binarize
binary_spike_count_matrix = data.binary_data_by_row(spike_count_matrix, threshold, max_active)

n_neurons, n_samples = spike_count_matrix.shape



print('Calculating Laplacians and Cochains...')
#Calculate Laplacians for each piece of data

st_ind_dict = data.build_ind_dict(binary_spike_count_matrix, max_simplex_dim)   #list of dictionaries that assign an index to each simplicial complex

full_binary_st = Laplacian.build_all_complexes(binary_spike_count_matrix, st_ind_dict, max_simplex_dim) #List of lists of dictionaries representing simplicial complexes (entries of outer list corespond to time bins; entries of inner lists correspond to simplicial dimensions)
bdry = Laplacian.build_boundaries(st_ind_dict)
Ups, Downs = Laplacian.build_ups_downs(bdry)
Up = [U.to_dense().clone().detach().to(device) for U in Ups]
Down = [D.to_dense().clone().detach().to(device) for D in Downs]

Up = data.Laplacian_power(Up, degree)#Returns list of lists of k-Hodge Laplacians raised to different powers
Down = data.Laplacian_power(Down, degree, ident=False)#Returns list of lists of k-Hodge Laplacians raised to different powers

Lap = []
Lap.append(Up[0])
for k in range(1, len(Up)):
	Lap.append(torch.cat([Up[k], Down[k-1]], dim=0))
Lap.append(Down[-1])

test_stop_idx = int(test_end_time*60 / (win_size))

n_edges = len(st_ind_dict[1])
n_triangles = len(st_ind_dict[2])
n_simp_list = [n_neurons, n_edges, n_triangles]
print('   Neurons, edges, triangles :', n_simp_list)


cochains = data.build_cochains(full_binary_st, spike_count_matrix, n_edges, n_triangles) #List of cochain tensors for each dimension. 

train_cochains = [C[...,test_stop_idx:] for C in cochains]  #slice out cochains used for training network


#prepare data for loading for training
if RNN:
	training_data = data.DatasetSCRNN(device, train_cochains, x[test_stop_idx:], y[test_stop_idx:], sequence_length)
elif intervals_per_sample==1:
	training_data = data.Dataset(device, train_cochains, x[test_stop_idx:], y[test_stop_idx:])
else:
	training_data = data.Dataset_gen(device, train_cochains, x[test_stop_idx:], y[test_stop_idx:], intervals_per_sample)



#########  Network  ###########
print('Building network...')
input_size = sum(n_simp_list[:(max_conv_dim+1)])

print('Flattened feature vector size:', input_size)

# load neural network 
if RNN: #SCRNN
	network = SCNN.SCNN_RNN(max_conv_dim, sc_layers, n_filters, sequence_length, n_simp_list, degree, Lap, input_size, rnn_layers, hidden_size, 2, dropout, \
	conv_activation, mlp_activation)
elif intervals_per_sample==1: #SCNN with only one time bin considered for each input
	network = SCNN.SCNN(max_conv_dim, sc_layers, n_filters, n_simp_list, degree, Lap, input_size, rnn_layers, nn_width, 2, dropout, \
	conv_activation, mlp_activation)
else: #SCNN with intervals_per_sample time bins considered for each input
	network = SCNN.SCNN_gen(max_conv_dim, sc_layers, n_filters, intervals_per_sample, n_simp_list, degree, Lap, input_size, rnn_layers, nn_width, 2, dropout, \
	conv_activation, mlp_activation)

# network = nn.DataParallel(network)
optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate) #Adam optimization
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) #learning rate scheduler

criterion = nn.MSELoss() #loss function

data_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)



print('Training network...')
train.train(network, device, data_loader, optimizer, criterion, scheduler, epochs) #train network



print('Plotting network prediction...')
#plot results
if RNN:
	plotting.plot_model_predictSCRNN(network, sequence_length, cochains, x, y, batch_size, test_stop_idx)
elif intervals_per_sample==1:
	plotting.plot_model_predict(network, cochains, x, y, batch_size)
else:
	plotting.plot_model_predict_gen(network, intervals_per_sample, cochains, x, y, batch_size, test_stop_idx)













