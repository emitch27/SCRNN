import numpy as np
import torch
import os, sys
from datetime import datetime



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
	lrs = []

	for step in range(epochs):
		cntr = 0
		losses_tmp = []

		##uncomment to check simplicial convolutional parameters at first and last epoch
		# if step==0 or step==epochs-1:
		# 	for l in range(model.sc_layers):
		# 		for m in range(model.max_dim+1):
		# 			print('SimpConv Weights for layer ' + str(l+1) + ' dim ' + str(m), \
		# 				[fw.grad for fw in model.simp_convlist[m][l].filter_weights])

		for i in data_loader:
			cntr += 1
			print(step, cntr)
			idx, sample, label = i


			label = label.to(device)
			# label = label.view(-1, 1).to(device)
			
			output = model(sample)


			loss = criterion(output, label)

			


			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses_tmp.append(loss.cpu().detach().numpy())
			epoch_loss = sum(losses_tmp) / len(losses_tmp)



		scheduler.step()
		print('Loss at step', step+1, ':', epoch_loss)

		lrs.append(optimizer.param_groups[0]['lr'])
		losses.append(epoch_loss)


		

	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	lrs_file = os.path.join(sys.path[0], "extracted/lrs")
	loss_file = os.path.join(sys.path[0], "extracted/loss")

	np.savetxt(lrs_file + dt + ".txt", np.array(lrs), fmt = '%1.18f', delimiter=',')
	np.savetxt(loss_file + dt + ".txt", np.array(losses), fmt = '%1.18f', delimiter=',')


#train RNN; saves learning rates and loss function values to files
def train_RNN(model, device, data_loader, optimizer, criterion, scheduler, epochs):
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
	lrs = []

	for step in range(epochs):
		cntr = 0
		losses_tmp = []

		for i in data_loader:
			cntr += 1
			print(step, cntr)
			idx, sample, label = i


			label = label.to(device)
			# label = label.view(-1, 1).to(device)
			
			output = model(sample.float())


			loss = criterion(output, label)

			


			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses_tmp.append(loss.cpu().detach().numpy())
			epoch_loss = sum(losses_tmp) / len(losses_tmp)



		scheduler.step()
		print('Loss at step', step+1, ':', epoch_loss)

		lrs.append(optimizer.param_groups[0]['lr'])
		losses.append(epoch_loss)


		

	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	lrs_file = os.path.join(sys.path[0], "extracted/lrs")
	loss_file = os.path.join(sys.path[0], "extracted/loss")

	np.savetxt(lrs_file + dt + ".txt", np.array(lrs), fmt = '%1.18f', delimiter=',')
	np.savetxt(loss_file + dt + ".txt", np.array(losses), fmt = '%1.18f', delimiter=',')






#train FFNN; saves learning rates and loss function values to files
def trainFFNN(model, device, data_loader, optimizer, criterion, scheduler, epochs):
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
	lrs = []

	for step in range(epochs):
		cntr = 0
		losses_tmp = []
		for i in data_loader:
			cntr += 1
			idx, sample, label = i


			label = label.to(device)

			output = model(sample.float()).to(device)


			loss = criterion(output, label)


			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses_tmp.append(loss.cpu().detach().numpy())
			epoch_loss = sum(losses_tmp) / len(losses_tmp)



		scheduler.step()
		print('Loss at step', step+1, ':', epoch_loss)

		lrs.append(optimizer.param_groups[0]['lr'])
		losses.append(epoch_loss)

	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	lrs_file = os.path.join(sys.path[0], "extracted/lrs")
	loss_file = os.path.join(sys.path[0], "extracted/loss")

	np.savetxt(lrs_file + dt + ".txt", np.array(lrs), fmt = '%1.18f', delimiter=',')
	np.savetxt(loss_file + dt + ".txt", np.array(losses), fmt = '%1.18f', delimiter=',')

