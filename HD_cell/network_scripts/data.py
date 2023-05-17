import gudhi as gd
import numpy as np
import torch
import scipy.sparse as sp



#Remove time bins with missing data (used for HD data)
def takeout_nan(count_matrix, angle_vec, times_vec):
	'''
	Inputs:
		count_matrix: spike count matrix
		angle_vec:    vector of ground truth head direction
		times_vec:    vector of time stamps for recordings

	Returns: all inputs with columns or vector elements featuring nan values removed
	'''
	ind = np.argwhere(~np.isnan(angle_vec))
	count_matrix = np.squeeze(count_matrix[:, ind])
	angle_vec = np.squeeze(angle_vec[ind])
	times_vec = np.squeeze(times_vec[ind])

	return count_matrix, angle_vec, times_vec




#Convert spike data to binary matrix
def binary_data_by_row(X, p, max_active):
	'''
	Inputs:
		X:          spike count matrix
		p:          threshold percentage
		max_active: max number of cells active in any time column (helps with computational cost)

	Returns: binarized spike count matrix
	'''
	binary_matrix = np.zeros(X.shape)
	p /= 100
	n_neurons, n_timebins = X.shape
	row_totals = np.sum(X, axis=1)

	for i in range(n_neurons):
		neuron = X[i,:]
		time_ind = neuron.argsort(axis=0)[::-1]
		threshold = p*row_totals[i]

		for j in range(n_timebins):
			ind = time_ind[:(j+1)]
			if np.sum(neuron[ind]) >= threshold:
				binary_matrix[i, ind] = 1.0
				break

	column_totals = np.sum(binary_matrix, axis=0)
	indx = np.argwhere(column_totals > max_active)
	for k in indx:
		activity = X[:,k] / row_totals
		most_active = activity.argsort(axis=0)[::-1]
		binary_matrix[most_active[max_active:],k] = 0.0

		
	return binary_matrix


#creates list of dictionaries that assign an index to each simplicial complex
def build_ind_dict(spike_count_matrix, max_dim):
	'''
	Inputs:
		spike_count_matrix: binarized spike count matrix
		max_dim:            maximum dimension of simplicial complex

	Returns: list of dictionaries that assign an index to each simplicial complex
	'''
	n_neurons, n_timebins = spike_count_matrix.shape

	st = gd.SimplexTree()
	for k in range(n_neurons):
		st.insert([k])
	for i in range(n_timebins):
		X_now = spike_count_matrix[:,i]
		spike_ind = np.argwhere(X_now==1)
		st.insert(spike_ind)

		st.set_dimension(max_dim)

	st_ind_dict = [dict() for _ in range(max_dim+1)]
	for sk_value in st.get_skeleton(max_dim):
		j = len(sk_value[0])
		if j<=max_dim+1:
			st_ind_dict[j - 1][frozenset(sk_value[0])] = len(st_ind_dict[j - 1])

	return st_ind_dict



#Create a simplicial complex for each time bin
def create_simplex(binary_X, max_dim, st_ind_dict):
	'''
	Inputs:
		binary_X:    column from binary spike count matrix
		max_dim:     maximum dimension of simplicial complex
		st_ind_dict: list of dictionaries that assign an index to every simplex across all time bins

	Returns: list of dictionaries that assign the appropriate index from st_ind_dict to each simplex present in column binary_X
	'''
	st = gd.SimplexTree()

	spike_ind = np.argwhere(binary_X==1)
	st.insert(spike_ind)
	st.set_dimension(max_dim+1)

	st_dict = [dict() for _ in range(max_dim+1)]
	for sk_value in st.get_skeleton(max_dim):
		j = len(sk_value[0])
		if frozenset(sk_value[0]) in list(st_ind_dict[j-1].keys()):
			st_dict[j - 1][frozenset(sk_value[0])] = st_ind_dict[j-1][frozenset(sk_value[0])]

	return st_dict



#raise upper and lower Laplacians to different powers
def Laplacian_power(Lap, degree, ident=True):
	'''
	Inputs:
		Lap:    list of laplacian matrices
		degree: highest degree to raise entries of Lap
		ident:  whether or not to raise Lap entries to 0th power (which gives identity matrix)

	Returns: list of lists of Lap raised to every power up to and including degree
	'''
	if ident:
		Lap_power = [list() for L in Lap]
		for k, L in enumerate(Lap):
			for i in range(degree+1):
				Lap_power[k].append(torch.matrix_power(L, i))

		for i in range(len(Lap_power)):
			Lap_power[i] = torch.stack(Lap_power[i], dim=0)
	else:
		Lap_power = [list() for L in Lap]
		for k, L in enumerate(Lap):
			for i in range(1, degree+1):
				Lap_power[k].append(torch.matrix_power(L, i))

		for i in range(len(Lap_power)):
			Lap_power[i] = torch.stack(Lap_power[i], dim=0)

	return Lap_power



#compute multi-correlation coefficient
def multicorrcoef(a, b, c):
	'''
	Inputs:
		a, b, c: all pearson correlation coefficients between three random variables
	
	Returns: multi-correlation coefficient
	'''
	m1 = np.sqrt((a**2 + c**2 - 2*a*b*c)/(1 - b**2))
	m2 = np.sqrt((a**2 + b**2 - 2*a*b*c)/(1 - c**2))
	m3 = np.sqrt((b**2 + c**2 - 2*a*b*c)/(1 - a**2))
	# print([m1, m2, m3])
	return min([m1, m2, m3])


#create cochains used for feature representation
def build_cochains(Sts, spike_count_matrix, n_edges, n_triangles):
	'''
	Inputs:
		Sts:                dictionary containing indices of each simplex
		spike_count_matrix: spike count matrix
		n_edges:            total number of 1-simplices
		n_triangles:        total number of 2-simplices

	Returns: List of cochain tensors (each entry corresponds to a different simplicial dimension)
	'''
	cochains = []
	ind1 = [0,1]
	ind2 = [0,2]
	ind3 = [1,2]

	#0-cochain
	cochains.append(spike_count_matrix.astype(np.float32))

	#1/2-cochain
	cochain1 = np.zeros((n_edges, spike_count_matrix.shape[1]))
	cochain2 = np.zeros((n_triangles, spike_count_matrix.shape[1]))

	corr = np.corrcoef(spike_count_matrix)
	
	for k, st_dict in enumerate(Sts):
		# print(k)
		cochain1_tmp = np.zeros(n_edges)
		cochain2_tmp = np.zeros(n_triangles)

		for simplex, idx in st_dict[1].items():
			cochain1_tmp[idx] = corr[tuple(simplex)].astype(np.float32)

		for simplex, idx in st_dict[2].items():
			simp = np.array(tuple(simplex))

			cochain2_tmp[idx] = multicorrcoef(corr[simp[ind1][0]][simp[ind1][1]].astype(np.float32), corr[simp[ind2][0]][simp[ind2][1]].astype(np.float32), corr[simp[ind3][0]][simp[ind3][1]].astype(np.float32))

		cochain1[:,k] = cochain1_tmp
		cochain2[:,k] = cochain2_tmp

	cochains.append(cochain1.astype(np.float32))
	cochains.append(cochain2.astype(np.float32))

	return cochains



#turn sparse coo matrices into dense torch tensors
def coo2tensor(A):
	'''
	Inputs:
		A: sparse coo matrix

	Returns: dense torch tensor of A
	'''
	assert(sp.isspmatrix_coo(A))
	idxs = torch.LongTensor(np.vstack((A.row, A.col)))
	vals = torch.FloatTensor(A.data)

	return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)




#builds dataset used for training SCNN with intervals_per_sample=1
class Dataset(torch.utils.data.Dataset):
	'''
	Inputs:
		device:   device that model and data is using for computing
		cochains: list of cochain tensors (each entry corresponds to a different simplicial dimension)
		labels:   vector of ground truth HD angles

	Returns: 
		idx:    index of input/output
		sample: input for NN
		label:  ground truth label used for computing loss
	'''
	def __init__(self, device, cochains, labels):
		self.device = device
		self.cochains = cochains
		self.labels = labels

	def __len__(self):
		return int(len(self.labels))

	def __getitem__(self, idx):
		sample = [torch.tensor(C[..., idx], device=self.device) for C in self.cochains]
		label = torch.tensor(self.labels[idx], dtype=torch.float32).squeeze()

		return idx, sample, label



#builds dataset used for training SCNN with intervals_per_sample>1
class Dataset_gen(torch.utils.data.Dataset):
	'''
	Inputs:
		device:               device that model and data is using for computing
		cochains:             list of cochain tensors (each entry corresponds to a different simplicial dimension)
		labels:               vector of ground truth HD angles
		intervals_per_sample: number of intervas (time bins) included for inputs into network

	Returns: 
		idx:    index of input/output
		sample: input for NN
		label:  ground truth label used for computing loss
	'''
	def __init__(self, device, cochains, labels, intervals_per_sample):
		self.device = device
		self.cochains = cochains
		self.labels = labels
		self.intps = intervals_per_sample

	def __len__(self):
		return int(len(self.labels)) - self.intps + 1

	def __getitem__(self, idx):
		sample = [torch.tensor(C[..., idx:idx+self.intps].T, device=self.device) for C in self.cochains]
		label = torch.tensor(self.labels[idx+self.intps-1], dtype=torch.float32).squeeze()

		return idx, sample, label

#builds dataset used for training SCRNN
class DatasetSCRNN(torch.utils.data.Dataset):
	'''
	Inputs:
		device:          device that model and data is using for computing
		cochains:        list of cochain tensors (each entry corresponds to a different simplicial dimension)
		labels:          vector of ground truth HD angles
		sequence_length: length of sequence to be included in RNN portion (automatically sets intervals_per_sample to same integer value)

	Returns: 
		idx:    index of input/output
		sample: input for NN
		label:  ground truth label used for computing loss
	'''
	def __init__(self, device, cochains, labels, sequence_length):
		self.device = device
		self.cochains = cochains
		self.labels = labels
		self.seq_length = sequence_length

	def __len__(self):
		return int(len(self.labels)) - self.seq_length + 1

	def __getitem__(self, idx):
		sample = [torch.tensor(C[..., idx:idx+self.seq_length].T, device=self.device) for C in self.cochains]
		label = torch.tensor(self.labels[idx+self.seq_length-1], dtype=torch.float32).squeeze()

		return idx, sample, label


#builds dataset used for training FFNN
class DatasetFFNN(torch.utils.data.Dataset):
	'''
	Inputs:
		device:             device that model and data is using for computing
		spike_count_matrix: spike count mtrix (individual columns are used as input into FFNN)
		labels:             vector of ground truth HD angles

	Returns: 
		idx:    index of input/output
		sample: input for NN
		label:  ground truth label used for computing loss 
	'''
	def __init__(self, spike_count_matrix, labels):
		self.spike_count_matrix = spike_count_matrix
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		sample = torch.tensor(self.spike_count_matrix[:,idx])
		label = torch.tensor(self.labels[idx])

		return idx, sample, label


#builds dataset used for training RNN
class DatasetRNN(torch.utils.data.Dataset):
	'''
	Inputs:
		device:             device that model and data is using for computing
		spike_count_matrix: spike count mtrix (individual columns are used as input into FFNN)
		labels:             vector of ground truth HD angles
		sequence_length:    length of sequence to be included in RNN

	Returns: 
		idx:    index of input/output
		sample: input for NN
		label:  ground truth label used for computing loss
	'''
	def __init__(self, spike_count_matrix, labels, sequence_length):
		self.spike_count_matrix = spike_count_matrix
		self.labels = labels
		self.seq_length = sequence_length

	def __len__(self):
		return int(len(self.labels)) - self.seq_length + 1

	def __getitem__(self, idx):
		sample = torch.tensor(self.spike_count_matrix[:,idx:idx+self.seq_length].T)
		label = torch.tensor(self.labels[idx+self.seq_length-1])

		return idx, sample, label




