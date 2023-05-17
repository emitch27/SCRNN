import numpy as np
import pickle
import torch.nn as nn
import torch
import time
from datetime import datetime
from scipy.linalg import block_diag
import data, Laplacian, plotting, SCNN, train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########  Parameters  ###########
win_size = 0.1   #window size used for time binning
data_interval = 2
end_time = 10    #minutes to include for total data
# end_time = 'end'   #if set to 'end', total data includes all data
test_end_time = 2   #minutes of total data are used for testing; the rest are used for training

intervals_per_sample = 3
sequence_length = 5   #length of input sequence used for RNN component

threshold = 1.5  #threshold parameter used in binarization step of pre-processing
max_simplex_dim = 2  #max dimension of simplices included in functional simplicial complex
max_active = 20   #max number of active cells in a time bin

epochs = 1
batch_size = 16
learning_rate = 0.001
dropout = 0.3

max_conv_dim = 1   #maximum dimension of convolutional filter used
sc_layers = 3   #number of simplicial convolutional layers
n_filters = 5   #number of filters per simplicial convolutional layer
degree = 1   #degree of simplicial filters
rnn_layers = 3   #number of recurrent layers
hidden_size = 200   #dimension of hidden components
conv_activation = 'relu'   #activation function used in simplicial convolutional layers
mlp_activation = 'relu'   #activation function used in recurrent layers


RNN=True


now = str(datetime.now())
dt = str(now[:10]+ '_' + str(now[11:])) #for tracking time of operations
print('start time :', dt)


#########  Data  ###########
print('Converting to training data...')
#Load data
if end_time=='end':
	file = open('../analyses/' + str(win_size*1000) + 'ms/interval' + str(data_interval) + 'mod1.p', 'rb')
	spike_data_dict = pickle.load(file)
	file.close()
	spike_count_matrix1 = spike_data_dict['count_matrix']
	# x = spike_data_dict['x']
	# y = spike_data_dict['y']
	x = spike_data_dict['x'] + 0.75
	y = spike_data_dict['y'] + 0.75

	file = open('../analyses/' + str(win_size*1000) + 'ms/interval' + str(data_interval) + 'mod2.p', 'rb')
	spike_data_dict = pickle.load(file)
	file.close()
	spike_count_matrix2 = spike_data_dict['count_matrix']

	file = open('../analyses/' + str(win_size*1000) + 'ms/interval' + str(data_interval) + 'mod3.p', 'rb')
	spike_data_dict = pickle.load(file)
	file.close()
	spike_count_matrix3 = spike_data_dict['count_matrix']

	
else:
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


test_stop_idx = int(test_end_time*60 / (win_size))
	

spike_count_matrix = np.vstack((spike_count_matrix1, spike_count_matrix2, spike_count_matrix3))

#Binarize
binary_spike_count_matrix1 = data.binary_data_by_row(spike_count_matrix1, threshold, max_active)   #generates binary spike count matrix for first module

binary_spike_count_matrix2 = data.binary_data_by_row(spike_count_matrix2, threshold, max_active)   #generates binary spike count matrix for second module

binary_spike_count_matrix3 = data.binary_data_by_row(spike_count_matrix3, threshold, max_active)   #generates binary spike count matrix for third module

n_neurons, n_samples = spike_count_matrix.shape



print('Calculating Laplacians and Cochains...')
#Calculate Laplacians for each piece of data

st_ind_dict1 = data.build_ind_dict(binary_spike_count_matrix1, max_simplex_dim)   #list of dictionaries that assign an index to each simplicial complex
full_binary_st1 = Laplacian.build_all_complexes(binary_spike_count_matrix1, st_ind_dict1, max_simplex_dim)    #List of lists of dictionaries representing simplicial complexes
bdry1 = Laplacian.build_boundaries(st_ind_dict1)   #Create incidence matrices used for filters
Ups1, Downs1 = Laplacian.build_ups_downs(bdry1)
Up1 = [U.to_dense().clone().detach().to(device) for U in Ups1]
Down1 = [D.to_dense().clone().detach().to(device) for D in Downs1]

Up1 = data.Laplacian_power(Up1, degree)#Returns list of lists of k-Hodge Laplacians raised to different powers
Down1 = data.Laplacian_power(Down1, degree, ident=False)#Returns list of lists of k-Hodge Laplacians raised to different powers


####  
Lap1 = []
Lap1.append(Up1[0])
for k in range(1, len(Up1)):
	Lap1.append(torch.cat([Up1[k], Down1[k-1]], dim=0))
Lap1.append(Down1[-1])








n_edges1 = len(st_ind_dict1[1])
n_triangles1 = len(st_ind_dict1[2])



st_ind_dict2 = data.build_ind_dict(binary_spike_count_matrix2, max_simplex_dim)   #list of dictionaries that assign an index to each simplicial complex
full_binary_st2 = Laplacian.build_all_complexes(binary_spike_count_matrix2, st_ind_dict2, max_simplex_dim)#List of lists of dictionaries representing simplicial complexes
bdry2 = Laplacian.build_boundaries(st_ind_dict2)
Ups2, Downs2 = Laplacian.build_ups_downs(bdry2)
Up2 = [U.to_dense().clone().detach().to(device) for U in Ups2]
Down2 = [D.to_dense().clone().detach().to(device) for D in Downs2]

Up2 = data.Laplacian_power(Up2, degree)#Returns list of lists of k-Hodge Laplacians raised to different powers
Down2 = data.Laplacian_power(Down2, degree, ident=False)#Returns list of lists of k-Hodge Laplacians raised to different powers

Lap2 = []
Lap2.append(Up2[0])
for k in range(1, len(Up2)):
	Lap2.append(torch.cat([Up2[k], Down2[k-1]], dim=0))
Lap2.append(Down2[-1])



n_edges2 = len(st_ind_dict2[1])
n_triangles2 = len(st_ind_dict2[2])





st_ind_dict3 = data.build_ind_dict(binary_spike_count_matrix3, max_simplex_dim)   #list of dictionaries that assign an index to each simplicial complex
full_binary_st3 = Laplacian.build_all_complexes(binary_spike_count_matrix3, st_ind_dict3, max_simplex_dim)#List of lists of dictionaries representing simplicial complexes
bdry3 = Laplacian.build_boundaries(st_ind_dict3)
Ups3, Downs3 = Laplacian.build_ups_downs(bdry3)
Up3 = [U.to_dense().clone().detach().to(device) for U in Ups3]
Down3 = [D.to_dense().clone().detach().to(device) for D in Downs3]

Up3 = data.Laplacian_power(Up3, degree)#Returns list of lists of k-Hodge Laplacians raised to different powers
Down3 = data.Laplacian_power(Down3, degree, ident=False)#Returns list of lists of k-Hodge Laplacians raised to different powers

Lap3 = []
Lap3.append(Up3[0])
for k in range(1, len(Up3)):
	Lap3.append(torch.cat([Up3[k], Down3[k-1]], dim=0))
Lap3.append(Down3[-1])



n_edges3 = len(st_ind_dict3[1])
n_triangles3 = len(st_ind_dict3[2])




print('edges 1: ', n_edges1)
print('edges 2: ', n_edges2)
print('edges 3: ', n_edges3)



print('triangles 1: ', n_triangles1)
print('triangles 2: ', n_triangles2)
print('triangles 3: ', n_triangles3)


n_edges = n_edges1 + n_edges2 + n_edges3
n_triangles = n_triangles1 + n_triangles2 + n_triangles3

n_simp_list = [n_neurons, n_edges, n_triangles]
print('   Neurons, edges, triangles :', n_simp_list)


Lap = []
for j in range(len(Lap1)):
	Lap_j = torch.empty(Lap1[j].size()[0], n_simp_list[j], n_simp_list[j])
	for k in range(Lap1[j].size()[0]):
		Lap_j[k,...] = torch.from_numpy(block_diag(Lap1[j][k,...].cpu().detach().numpy(), Lap2[j][k,...].cpu().detach().numpy(), Lap3[j][k,...].cpu().detach().numpy()))
	Lap.append(Lap_j.to(device))



cochains1 = data.build_cochains(full_binary_st1, spike_count_matrix1, n_edges1, n_triangles1) #List of cochain tensors for each dimension. 
cochains2 = data.build_cochains(full_binary_st2, spike_count_matrix2, n_edges2, n_triangles2) #List of cochain tensors for each dimension. 
cochains3 = data.build_cochains(full_binary_st3, spike_count_matrix3, n_edges3, n_triangles3) #List of cochain tensors for each dimension. 


cochains = []
for k in range(len(cochains1)):
	cochains.append(np.vstack((cochains1[k], cochains2[k], cochains3[k])))



train_cochains = [C[...,test_stop_idx:] for C in cochains]


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

criterion = nn.MSELoss()   #loss function

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






