import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




#initialize network weights
def init_weights(m):
    '''
    Inputs:
        m: network parameters
    '''
    if type(m) == nn.Linear or type(m) ==nn.Parameter:
        nn.init.xavier_uniform_(m.weight)


#define first layer simplicial convolution
class simplicial_conv_in(nn.Module):
	def __init__(self, n_filters, degree, activation, Lap_tensor, device):
		super().__init__()

		self.degree = degree
		self.n_filters = n_filters
		self.L = Lap_tensor
		self.device = device

		self.filter_weights = []
		for _ in range(self.n_filters):
			self.filter_weights.append(nn.Parameter(torch.normal(mean=torch.zeros(self.L.size()[0]), std=.01*torch.ones(self.L.size()[0])).to(device), requires_grad=True))
		
		if activation == 'tanh':
			self.activation = torch.tanh
		elif activation == 'relu':
			self.activation = torch.relu
		elif activation == 'leaky_relu':
			self.activation = nn.LeakyReLU(0.01)
		elif activation == 'sigmoid':
			self.activation = torch.sigmoid



	def forward(self, x):
		outputs = list()# should ultimately output list of size n_filters 

		for k in range(self.n_filters):
			weight_matrix = torch.mul(torch.eye(self.L.size()[0], device=self.device), self.filter_weights[k]) #identity matrix times weights used for each degree
			batches = []
			for i in range(x.size()[0]):
				batches.append(torch.matmul(self.L, x[i,:]))
			Lx = torch.stack(batches, dim=0)
			weighted_tensor = torch.matmul(weight_matrix, Lx) #multiply each term by the weight
			filter_out = self.activation(torch.sum(weighted_tensor, dim=1)) #sum each term and feed to activation

			outputs.append(filter_out) #sum each term and append to output list

		return outputs



#define "middle" (not first or last) layer simplicial convolution
class simplicial_conv(nn.Module):
	def __init__(self, n_filters, degree, activation, Lap_tensor, device):
		super().__init__()

		self.n_filters = n_filters
		self.L = Lap_tensor
		self.device = device

		self.filter_weights = []
		for _ in range(self.n_filters):
			self.filter_weights.append(nn.Parameter(torch.normal(mean=torch.zeros(self.L.size()[0]), std=.01*torch.ones(self.L.size()[0])).to(device), requires_grad=True))
		
		if activation == 'tanh':
			self.activation = torch.tanh
		elif activation == 'relu':
			self.activation = torch.relu
		elif activation == 'leaky_relu':
			self.activation = nn.LeakyReLU(0.01)
		elif activation == 'sigmoid':
			self.activation = torch.sigmoid


	def forward(self, x_list):
		outputs = list()# should ultimately output list of size n_filters 

		for x in x_list:
			outputs_x = list()
			for k in range(self.n_filters):
				weight_matrix = torch.mul(torch.eye(self.L.size()[0], device=self.device), self.filter_weights[k]) #identity matrix times weights used for each degree
				batches = []
				for i in range(x.size()[0]):
					batches.append(torch.matmul(self.L, x[i,:]))
				Lx = torch.stack(batches, dim=0)
				weighted_tensor = torch.matmul(weight_matrix, Lx) #multiply each term by the weight
				filter_out = self.activation(torch.sum(weighted_tensor, dim=1)) #sum each term and feed to activation

				outputs_x.append(filter_out) #sum each term and append to output list
			outputs.append(torch.sum(torch.stack(outputs_x, dim=0), dim=0))

		return outputs



#define last simplicial convolutional layer
class simplicial_conv_out(nn.Module):
	def __init__(self, n_filters, degree, activation, Lap_tensor, device):
		super().__init__()

		self.n_filters = n_filters
		self.L = Lap_tensor
		self.device = device

		self.filter_weights = []
		for _ in range(self.n_filters):
			self.filter_weights.append(nn.Parameter(torch.normal(mean=torch.zeros(self.L.size()[0]), std=.01*torch.ones(self.L.size()[0])).to(device), requires_grad=True))
		
		if activation == 'tanh':
			self.activation = torch.tanh
		elif activation == 'relu':
			self.activation = torch.relu
		elif activation == 'leaky_relu':
			self.activation = nn.LeakyReLU(0.01)
		elif activation == 'sigmoid':
			self.activation = torch.sigmoid


	def forward(self, x_list):
		outputs = list()# should ultimately output list of size n_filters 

		for x in x_list:
			outputs_x = list()
			for k in range(self.n_filters):
				weight_matrix = torch.mul(torch.eye(self.L.size()[0], device=self.device), self.filter_weights[k]) #identity matrix times weights used for each degree
				batches = []
				for i in range(x.size()[0]):
					batches.append(torch.matmul(self.L, x[i,:]))
				Lx = torch.stack(batches, dim=0)
				weighted_tensor = torch.matmul(weight_matrix, Lx) #multiply each term by the weight
				filter_out = self.activation(torch.sum(weighted_tensor, dim=1)) #sum each term and feed to activation

				outputs_x.append(filter_out) #sum each term and append to output list
			outputs.append(torch.sum(torch.stack(outputs_x, dim=0), dim=0))


		return torch.sum(torch.stack(outputs, dim=0), dim=0)


#define simplicial convolutional layer for network with only one such layer
class simplicial_conv_indie(nn.Module):
	def __init__(self, n_filters, degree, activation, Lap_tensor, device):
		super().__init__()

		self.degree = degree
		self.n_filters = n_filters
		self.L = Lap_tensor
		self.device = device

		self.filter_weights = []
		for _ in range(self.n_filters):
			self.filter_weights.append(nn.Parameter(torch.normal(mean=torch.zeros(self.L.size()[0]), std=.01*torch.ones(self.L.size()[0])).to(device), requires_grad=True))
		
		if activation == 'tanh':
			self.activation = torch.tanh
		elif activation == 'relu':
			self.activation = torch.relu
		elif activation == 'leaky_relu':
			self.activation = nn.LeakyReLU(0.01)
		elif activation == 'sigmoid':
			self.activation = torch.sigmoid


	def forward(self, x):
		outputs = list()# should ultimately output list of size n_filters 

		for k in range(self.n_filters):
			weight_matrix = torch.mul(torch.eye(self.L.size()[0], device=self.device), self.filter_weights[k]) #identity matrix times weights used for each degree
			batches = []
			for i in range(x.size()[0]):
				batches.append(torch.matmul(self.L, x[i,:]))
			Lx = torch.stack(batches, dim=0)
			weighted_tensor = torch.matmul(weight_matrix, Lx) #multiply each term by the weight
			filter_out = self.activation(torch.sum(weighted_tensor, dim=1)) #sum each term and feed to activation
			outputs.append(filter_out) #sum each term and append to output list

		return torch.sum(torch.stack(outputs, dim=0), dim=0)



#define FFNN
class MLP(nn.Module):
	def __init__(self, in_size, nn_layers, nn_width, output_size, dropout, activation):
		super().__init__()

		self.linear_in = nn.Linear(in_size, nn_width[0])
		self.linear = nn.ModuleList()
		for i in range(nn_layers-1):
			self.linear.append(nn.Linear(nn_width[i], nn_width[i+1]))

		self.linear_out = nn.Linear(nn_width[-1], output_size)
		self.dropout = nn.Dropout(dropout)
		self.batch_normMLP = nn.BatchNorm1d(nn_width)

		if activation == 'tanh':
			self.activation = torch.tanh
		elif activation == 'relu':
			self.activation = torch.relu
		elif activation == 'leaky_relu':
			self.activation = nn.LeakyReLU
		elif activation == 'sigmoid':
			self.activation = torch.sigmoid

		self.apply(init_weights)

	def forward(self, x):
		x = self.activation(self.linear_in(x))
		# x = self.batch_normMLP(x)
		x = self.dropout(x)
		for layer in self.linear:
			x_temp = self.activation(layer(x))			
			# x_temp = self.batch_normMLP(x_temp)
			x_temp = self.dropout(x_temp)
			x = x_temp
		x = self.linear_out(x)
		
		return x



#define SCNN with intervals_per_sample=1
class SCNN(nn.Module):
	def __init__(self, max_dim, sc_layers, n_filters, n_simp_list, degree, Laplacians, in_size, nn_layers, nn_width, output_size, dropout, conv_activation, mlp_activation):
		super().__init__()

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.max_dim = max_dim
		self.simp_convlist = [nn.ModuleList() for _ in range(self.max_dim + 1)]
		# self.batch_norm_list = [nn.ModuleList() for _ in range(self.max_dim + 1)]

		self.Laplacians = Laplacians
		self.sc_layers = sc_layers

		if sc_layers==1:
			for i in range(self.max_dim + 1):
				self.simp_convlist[i].append(simplicial_conv_indie(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
		else:
			for i in range(self.max_dim + 1):
				self.simp_convlist[i].append(simplicial_conv_in(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
				for _ in range(sc_layers - 2):
					self.simp_convlist[i].append(simplicial_conv(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
					# self.batch_norm_list[i].append(nn.BatchNorm2d(1))

				self.simp_convlist[i].append(simplicial_conv_out(n_filters, degree, conv_activation, self.Laplacians[i], self.device))

		self.MLP = MLP(in_size, nn_layers, nn_width, output_size, dropout, mlp_activation)


		self.to(self.device)

	def forward(self, xs):
		output_list = list()
		for i in range(self.max_dim + 1):
			x = xs[i]
			for k, layer in enumerate(self.simp_convlist[i]):
				x_temp = layer(x)
				# x_temp = self.batch_norm_list[i][k](x_temp)
				x = x_temp

			output_list.append(x)
		concat_output = torch.cat(output_list, 1)
		

		return self.MLP(concat_output)



#define SCNN with intervals_per_sample>1
class SCNN_gen(nn.Module):
	def __init__(self, max_dim, sc_layers, n_filters, sequence_length, n_simp_list, degree, Laplacians, in_size, nn_layers, nn_width, output_size, dropout, conv_activation, mlp_activation):
		super().__init__()

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.max_dim = max_dim
		self.seq_length = sequence_length
		self.simp_convlist = [nn.ModuleList() for _ in range(self.max_dim + 1)]
		# self.batch_norm_list = [nn.ModuleList() for _ in range(self.max_dim + 1)]

		self.Laplacians = Laplacians
		self.sc_layers = sc_layers

		if sc_layers==1:
			for i in range(self.max_dim + 1):
				self.simp_convlist[i].append(simplicial_conv_indie(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
		else:
			for i in range(self.max_dim + 1):
				self.simp_convlist[i].append(simplicial_conv_in(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
				for _ in range(sc_layers - 2):
					self.simp_convlist[i].append(simplicial_conv(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
					# self.batch_norm_list[i].append(nn.BatchNorm2d(1))

				self.simp_convlist[i].append(simplicial_conv_out(n_filters, degree, conv_activation, self.Laplacians[i], self.device))

		self.MLP = MLP(in_size, nn_layers, nn_width, output_size, dropout, mlp_activation)


		self.to(self.device)


	def forward(self, xs):
		big_out_list = list()
		for k in range(self.seq_length):
			output_list = list()
			for i in range(self.max_dim + 1):
				x = xs[i][:,k,:]
				for k, layer in enumerate(self.simp_convlist[i]):
					x_temp = layer(x)
					# x_temp = self.batch_norm_list[i][k](x_temp)
					x = x_temp

				output_list.append(x)
			big_out_list.append(torch.cat(output_list, 1))
		out = torch.stack(big_out_list, 1)
		out = torch.sum(out, axis=1)

		

		return self.MLP(out)


#define SCRNN
class SCNN_RNN(nn.Module):
	def __init__(self, max_dim, sc_layers, n_filters, sequence_length, n_simp_list, degree, Laplacians, in_size, nn_layers, nn_width, output_size, dropout, conv_activation, rnn_activation):
		super().__init__()

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.max_dim = max_dim
		self.seq_length = sequence_length
		# self.simp_convlist = [nn.ModuleList() for _ in range(self.max_dim + 1)]
		self.simp_convlist = nn.ModuleList(nn.ModuleList() for _ in range(self.max_dim + 1))
		# self.batch_norm_list = [nn.ModuleList() for _ in range(self.max_dim + 1)]

		self.Laplacians = Laplacians
		self.sc_layers = sc_layers

		if sc_layers==1:
			for i in range(self.max_dim + 1):
				self.simp_convlist[i].append(simplicial_conv_indie(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
		else:
			for i in range(self.max_dim + 1):
				self.simp_convlist[i].append(simplicial_conv_in(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
				for _ in range(sc_layers - 2):
					self.simp_convlist[i].append(simplicial_conv(n_filters, degree, conv_activation, self.Laplacians[i], self.device))
					# self.batch_norm_list[i].append(nn.BatchNorm2d(1))

				self.simp_convlist[i].append(simplicial_conv_out(n_filters, degree, conv_activation, self.Laplacians[i], self.device))

		if nn_layers==1:
			self.RNN = nn.RNN(input_size=in_size, hidden_size=nn_width, num_layers=nn_layers, nonlinearity=rnn_activation, batch_first=True)
		else:
			self.RNN = nn.RNN(input_size=in_size, hidden_size=nn_width, num_layers=nn_layers, nonlinearity=rnn_activation, batch_first=True, dropout=dropout)

		self.linear_out = nn.Linear(nn_width, output_size)


		self.to(self.device)

	def forward(self, xs):
		big_out_list = list()
		for k in range(self.seq_length):
			output_list = list()
			for i in range(self.max_dim + 1):
				x = xs[i][:,k,:]
				for k, layer in enumerate(self.simp_convlist[i]):
					x_temp = layer(x)
					# x_temp = self.batch_norm_list[i][k](x_temp)
					x = x_temp

				output_list.append(x)
			big_out_list.append(torch.cat(output_list, 1))
		concat_output = torch.stack(big_out_list, 1)
		out, _ = self.RNN(concat_output)
		out = self.linear_out(out)[:,-1,:]

		return out


#define RNN
class RNN(nn.Module):
	def __init__(self, device, in_size, out_size, nn_width, nn_layers, rnn_activation, dropout):
		super().__init__()

		if nn_layers==1:
			self.RNN = nn.RNN(input_size=in_size, hidden_size=nn_width, num_layers=nn_layers, nonlinearity=rnn_activation, batch_first=True)
		else:
			self.RNN = nn.RNN(input_size=in_size, hidden_size=nn_width, num_layers=nn_layers, nonlinearity=rnn_activation, batch_first=True, dropout=dropout)


		self.linear_out = nn.Linear(nn_width, out_size)
		self.device = device

		self.to(self.device)

	def forward(self, x):
		out, _ = self.RNN(x)
		out = self.linear_out(out)[:,-1,:]

		return out











