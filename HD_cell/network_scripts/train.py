import numpy as np
import torch
import torch.nn as nn



#train simplicial convolutional network; saves learning rates and loss function values to files
def train(model, device, data_loader, optimizer, criterion, scheduler, epochs):
	'''
	Inputs:
		model:       neural network being trained
		device:      device on which data and network are stored and using for computation
		data_loader: torch data loader holding all training data
		optimizer:   optimizer being used on network parameters
		criterion:   loss function
		scheduler:   learning rate scheduller
		epochs:      number of epochs for training
	'''
	model.train()

	losses = []

	for step in range(epochs):
		cntr = 0
		# print('epoch:', cntr)
		# for name, param in model.named_parameters():
		# 	print(name, param.data)
		# 	print('requires grad:', param.requires_grad)
		losses_tmp = []
		for i in data_loader:
			cntr += 1
			idx, sample, label = i

			label = label.view(-1, 1).to(device)
			# label = label.unsqueeze(1).to(device)

			
			# print(label.size())


			output = model(sample).to(device)
			# print(output)

			loss = criterion(output, label)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses_tmp.append(loss.cpu().detach().numpy())
			epoch_loss = sum(losses_tmp) / len(losses_tmp)
		print('Loss at step', step+1, ':', epoch_loss)
		losses.append(epoch_loss)

		scheduler.step()
		##uncomment to check model gradients
		# for k in range(len(model.simp_convlist[0][0].filter_weights)):
		# 	print('simpConv weight gradients:', model.simp_convlist[0][0].filter_weights[k].grad)
		# print('MLP input weight gradients:', model.MLP.linear_in.weight.grad)
		# print('MLP second layer weight gradients:', model.MLP.linear[0].weight.grad)
		# for name, param in model.named_parameters():
		# 	print(name, param.data)
		# 	print('requires grad:', param.requires_grad)



#train FFNN; saves learning rates and loss function values to files
def trainFFNN(model, device, data_loader, optimizer, criterion, epochs):
	'''
	Inputs:
		model:       neural network being trained
		device:      device on which data and network are stored and using for computation
		data_loader: torch data loader holding all training data
		optimizer:   optimizer being used on network parameters
		criterion:   loss function
		scheduler:   learning rate scheduller
		epochs:      number of epochs for training
	'''
	model.train()

	losses = []

	for step in range(epochs):
		losses_tmp = []
		for i in data_loader:
			idx, sample, label = i

			label = label.to(device)




			output = model(sample.float()).to(device)
			loss = criterion(output, label.unsqueeze(1))
			# loss = criterion(output, label)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()


			losses_tmp.append(loss.cpu().detach().numpy())
			epoch_loss = sum(losses_tmp) / len(losses_tmp)
		print('Loss at step', step+1, ':', epoch_loss)
		losses.append(epoch_loss)





#train RNN; saves learning rates and loss function values to files
def trainRNN(model, device, data_loader, optimizer, criterion, scheduler, epochs):
	'''
	Inputs:
		model:       neural network being trained
		device:      device on which data and network are stored and using for computation
		data_loader: torch data loader holding all training data
		optimizer:   optimizer being used on network parameters
		criterion:   loss function
		scheduler:   learning rate scheduller
		epochs:      number of epochs for training
	'''
	model.train()

	losses = []

	for step in range(epochs):
		cntr = 0
		losses_tmp = []
		for i in data_loader:
			cntr += 1
			idx, sample, label = i

			label = label.view(-1, 1).to(device)


			output = model(sample.float()).to(device)

			loss = criterion(output, label)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses_tmp.append(loss.cpu().detach().numpy())
			epoch_loss = sum(losses_tmp) / len(losses_tmp)
		print('Loss at step', step+1, ':', epoch_loss)
		losses.append(epoch_loss)

		scheduler.step()



#compute median absolute error and average absolute error 
def MAE(output, label):
	'''
	Inputs:
		output: network output across all time bins
		label: ground truth HD angles across all time bins

	Returns:
		mae: median absolute error
		aae: average absolute error
	'''
	diff = np.abs(output - label)
	diff[diff > 180.0] = np.abs(360  - diff[diff > 180.0])

	return np.median(diff), np.mean(diff)








