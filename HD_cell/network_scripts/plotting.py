import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import train
import os, sys
from datetime import datetime






#plot model prediction and compute perormance measurements of SCNN with seq_length=1
def plot_model_predict(model, cochains, angles, times, test_stop_idx):
	'''
	Inputs:
		model:         SCNN that has been trained
		cochains:      list of cochain tensors for each dimension
		angles:        ground truth HD angles
		times:         times (in seconds) of HD recordings
		test_stop_idx: index for slicing test data and training data from total data

	'''
	n_times = angles.size
	cochains = [torch.tensor(C.T, device=model.device) for C in cochains]

	model.eval()
	predict = model(cochains).detach().numpy().squeeze()


	indx = []
	# angles = (np.degrees(angles) + 180) % 360
	angles = np.degrees(angles)
	predict = np.degrees(predict)
	times = times-times[0]

	SCNNerror = np.abs(predict - angles)
	SCNNerror[SCNNerror > 180.0] = np.abs(360  - SCNNerror[SCNNerror > 180.0])
	
	SCNNcata = np.argwhere(SCNNerror > 90)

	print('SCNN Catastrophic:', SCNNcata.size)


	train_MAE, train_AAE = train.MAE(predict[test_stop_idx:], angles[test_stop_idx:])
	print('train MAE:',  train_MAE)
	print('train AAE:',  train_AAE)

	test_MAE, test_AAE = train.MAE(predict[:test_stop_idx], angles[:test_stop_idx])
	print('test MAE:',  test_MAE)
	print('test AAE:',  test_AAE)

	total_MAE, total_AAE = train.MAE(predict, angles)
	print('total MAE:',  total_MAE)
	print('total AAE:',  total_AAE)

	for i in range(n_times - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times, angles, linewidth=1.5, color='blue', label='true')
		plt.plot(times, predict, linewidth=1.5, color='red', label='decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Decoded Head Angle', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	predict_file = os.path.join(sys.path[0], "extracted/model_prediction")
	np.savetxt(predict_file + dt + ".txt", predict, fmt = '%1.18f', delimiter=',')


	end_time = 1200
	indx = []
	for i in range(end_time - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times[:end_time], angles[:end_time], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:end_time], predict[:end_time], linewidth=1.5, color='red', label='decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Decoded Head Angle', fontsize=16)
	plt.legend()
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	plt.figure(figsize=(12,5))

	plt.plot(times, SCNNerror, linewidth=1.5, color='red', label='SCNN')

	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Absolute Error Decoded Head Angle', fontsize=16)
	# plt.legend()

	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/errorplot" + dt + ".pdf"))
	# plt.show()
	plt.close()


	MAE_file = open(os.path.join(sys.path[0], "extracted/trainMAE" + dt + ".txt"), "w")
	MAE_file.write(str(train_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/trainAAE" + dt + ".txt"), "w")
	AAE_file.write(str(train_AAE))
	AAE_file.close()

	MAE_file = open(os.path.join(sys.path[0], "extracted/testMAE" + dt + ".txt"), "w")
	MAE_file.write(str(test_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/testAAE" + dt + ".txt"), "w")
	AAE_file.write(str(test_AAE))
	AAE_file.close()



#plot model prediction and compute perormance measurements of SCNN with multiple time bins considered for each input
def plot_model_predict_gen(model, seq_length, cochains, angles, times, test_stop_idx):
	'''
	Inputs:
		model:         SCNN that has been trained
		seq_length:    how many time bins are used for individual inpus into network
		cochains:      list of cochain tensors for each dimension
		angles:        ground truth HD angles
		times:         times (in seconds) of HD recordings
		test_stop_idx: index for slicing test data and training data from total data

	'''
	n_times = int(angles.size - seq_length + 1)
	angles = angles[seq_length-1:]
	times = times[seq_length-1:]


	model.eval()

	predict = []

	for k in range(n_times):
		cochains_tmp = [torch.tensor(C[..., k:k+seq_length].T, device=model.device).unsqueeze(0) for C in cochains]
		pred = model(cochains_tmp).cpu().detach().numpy().squeeze()
		predict.append(pred)

	predict = np.array(predict).squeeze()

	indx = []
	# angles = (np.degrees(angles) + 180) % 360
	angles = np.degrees(angles)
	predict = np.degrees(predict)
	times = times-times[0]

	SCNNerror = np.abs(predict - angles)
	SCNNerror[SCNNerror > 180.0] = np.abs(360  - SCNNerror[SCNNerror > 180.0])
	
	SCNNcata = np.argwhere(SCNNerror > 90)

	print('SCNN Catastrophic:', SCNNcata.size)


	train_MAE, train_AAE = train.MAE(predict[test_stop_idx:], angles[test_stop_idx:])
	print('train MAE:',  train_MAE)
	print('train AAE:',  train_AAE)

	test_MAE, test_AAE = train.MAE(predict[:test_stop_idx], angles[:test_stop_idx])
	print('test MAE:',  test_MAE)
	print('test AAE:',  test_AAE)

	total_MAE, total_AAE = train.MAE(predict, angles)
	print('total MAE:',  total_MAE)
	print('total AAE:',  total_AAE)

	for i in range(n_times - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times, angles, linewidth=1.5, color='blue', label='true')
		plt.plot(times, predict, linewidth=1.5, color='red', label='decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Decoded Head Angle', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	predict_file = os.path.join(sys.path[0], "extracted/model_prediction")
	np.savetxt(predict_file + dt + ".txt", predict, fmt = '%1.18f', delimiter=',')


	end_time = 1200
	indx = []
	for i in range(end_time - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times[:end_time], angles[:end_time], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:end_time], predict[:end_time], linewidth=1.5, color='red', label='decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Decoded Head Angle', fontsize=16)
	plt.legend()
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	plt.figure(figsize=(12,5))

	plt.plot(times, SCNNerror, linewidth=1.5, color='red', label='SCNN')

	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Absolute Error Decoded Head Angle', fontsize=16)
	# plt.legend()

	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/errorplot" + dt + ".pdf"))
	# plt.show()
	plt.close()


	MAE_file = open(os.path.join(sys.path[0], "extracted/trainMAE" + dt + ".txt"), "w")
	MAE_file.write(str(train_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/trainAAE" + dt + ".txt"), "w")
	AAE_file.write(str(train_AAE))
	AAE_file.close()

	MAE_file = open(os.path.join(sys.path[0], "extracted/testMAE" + dt + ".txt"), "w")
	MAE_file.write(str(test_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/testAAE" + dt + ".txt"), "w")
	AAE_file.write(str(test_AAE))
	AAE_file.close()



#plot model prediction and compute perormance measurements of SCNN with multiple time bins considered for each input
def plot_model_predictSCRNN(model, seq_length, cochains, angles, times, test_stop_idx):
	'''
	Inputs:
		model:         SCRNN that has been trained
		seq_length:    how many time bins are used for individual inpus into network
		cochains:      list of cochain tensors for each dimension
		angles:        ground truth HD angles
		times:         times (in seconds) of HD recordings
		test_stop_idx: index for slicing test data and training data from total data

	'''
	n_times = int(angles.size - seq_length + 1)
	test_stop_idx -= seq_length
	times = times[seq_length-1:]


	model.eval()

	predict = []
	for k in range(n_times):
		# cochains_tmp = [C[..., k:k+seq_length].clone().detach().T.unsqueeze(0).to(model.device) for C in cochains]
		cochains_tmp = [torch.tensor(C[..., k:k+seq_length].T, device=model.device).unsqueeze(0) for C in cochains]
		predict.append(model(cochains_tmp).detach().numpy().squeeze())



	indx = []
	# angles = (np.degrees(angles) + 180) % 360
	angles = np.degrees(angles[seq_length-1:])
	predict = np.degrees(np.array(predict))
	times = times-times[0]

	SCNNerror = np.abs(predict - angles)
	SCNNerror[SCNNerror > 180.0] = np.abs(360  - SCNNerror[SCNNerror > 180.0])
	
	SCNNcata = np.argwhere(SCNNerror > 90)

	print('SCRNN Catastrophic:', SCNNcata.size)


	train_MAE, train_AAE = train.MAE(predict[test_stop_idx:], angles[test_stop_idx:])
	print('train MAE:',  train_MAE)
	print('train AAE:',  train_AAE)

	test_MAE, test_AAE = train.MAE(predict[:test_stop_idx], angles[:test_stop_idx])
	print('test MAE:',  test_MAE)
	print('test AAE:',  test_AAE)

	total_MAE, total_AAE = train.MAE(predict, angles)
	print('total MAE:',  total_MAE)
	print('total AAE:',  total_AAE)

	for i in range(n_times - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times, angles, linewidth=1.5, color='blue', label='true')
		plt.plot(times, predict, linewidth=1.5, color='red', label='decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Decoded Head Angle', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/SCRNNmodel_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	predict_file = os.path.join(sys.path[0], "extracted/SCRNNmodel_prediction")
	np.savetxt(predict_file + dt + ".txt", predict, fmt = '%1.18f', delimiter=',')


	end_time = 1200
	indx = []
	for i in range(end_time - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times[:end_time], angles[:end_time], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:end_time], predict[:end_time], linewidth=1.5, color='red', label='decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Decoded Head Angle', fontsize=16)
	plt.legend()
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/SCRNNmodel_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	plt.figure(figsize=(12,5))

	plt.plot(times, SCNNerror, linewidth=1.5, color='red', label='SCNN')

	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Absolute Error Decoded Head Angle', fontsize=16)
	# plt.legend()

	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/SCRNNerrorplot" + dt + ".pdf"))
	# plt.show()
	plt.close()


	MAE_file = open(os.path.join(sys.path[0], "extracted/SCRNNtrainMAE" + dt + ".txt"), "w")
	MAE_file.write(str(train_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/SCRNNtrainAAE" + dt + ".txt"), "w")
	AAE_file.write(str(train_AAE))
	AAE_file.close()

	MAE_file = open(os.path.join(sys.path[0], "extracted/SCRNNtestMAE" + dt + ".txt"), "w")
	MAE_file.write(str(test_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/SCRNNtestAAE" + dt + ".txt"), "w")
	AAE_file.write(str(test_AAE))
	AAE_file.close()



#plot model prediction and compute perormance measurements of FFNN
def plot_model_predictFFNN(model, spike_count_matrix, angles, times, test_stop_idx):
	'''
	Inputs:
		model:              FFNN that has been trained
		spike_count_matrix: matrix of spike counts (rows indicate neurons; columns are time bins)
		angles:             ground truth HD angles
		times:              times (in seconds) of HD recordings
		test_stop_idx:      index for slicing test data and training data from total data

	'''
	model.eval()
	# input_tens = torch.tensor(np.expand_dims(spike_count_matrix, axis=2)).permute(1, 0, 2)
	input_tens = torch.tensor(spike_count_matrix.T)
	pred_tens = model(input_tens.float())

	predict = pred_tens.detach().numpy().squeeze()

	
	indx = []
	# angles = (np.degrees(angles) + 180) % 360
	angles = np.degrees(angles)
	predict = np.degrees(predict)
	times = times-times[0]

	FFNNerror = np.abs(predict - angles)
	FFNNerror[FFNNerror > 180.0] = np.abs(360  - FFNNerror[FFNNerror > 180.0])
	
	FFNNcata = np.argwhere(FFNNerror > 90)

	print('FFNN Catastrophic:', FFNNcata.size)

	train_MAE, train_AAE = train.MAE(predict[test_stop_idx:], angles[test_stop_idx:])
	print('train MAE:',  train_MAE)
	print('train AAE:',  train_AAE)

	test_MAE, test_AAE = train.MAE(predict[:test_stop_idx], angles[:test_stop_idx])
	print('test MAE:',  test_MAE)
	print('test AAE:',  test_AAE)

	total_MAE, total_AAE = train.MAE(predict, angles)
	print('total MAE:',  total_MAE)
	print('total AAE:',  total_AAE)

	for i in range(angles.size - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='FFNN decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times, angles, linewidth=1.5, color='blue', label='true')
		plt.plot(times, predict, linewidth=1.5, color='red', label='FFNN decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('FFNN Decoded Head Angle', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/FFNNmodel_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	predict_file = os.path.join(sys.path[0], "extracted/FFNNmodel_prediction")
	np.savetxt(predict_file + dt + ".txt", predict, fmt = '%1.18f', delimiter=',')


	end_time = 1200
	indx = []
	for i in range(end_time - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='FFNN decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times[:end_time], angles[:end_time], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:end_time], predict[:end_time], linewidth=1.5, color='red', label='FFNN decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('FFNN Decoded Head Angle', fontsize=16)
	plt.legend()
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/FFNNmodel_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	plt.figure(figsize=(12,5))

	plt.plot(times, FFNNerror, linewidth=1.5, color='red', label='FFNN')

	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Absolute Error Decoded Head Angle', fontsize=16)
	# plt.legend()

	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/FFNNerrorplot" + dt + ".pdf"))
	# plt.show()
	plt.close()


	MAE_file = open(os.path.join(sys.path[0], "extracted/trainMAE" + dt + ".txt"), "w")
	MAE_file.write(str(train_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/trainAAE" + dt + ".txt"), "w")
	AAE_file.write(str(train_AAE))
	AAE_file.close()

	MAE_file = open(os.path.join(sys.path[0], "extracted/testMAE" + dt + ".txt"), "w")
	MAE_file.write(str(test_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/testAAE" + dt + ".txt"), "w")
	AAE_file.write(str(test_AAE))
	AAE_file.close()


#plot model prediction and compute perormance measurements of RNN
def plot_model_predictRNN(model, seq_length, spike_count_matrix, angles, times, test_stop_idx):
	'''
	Inputs:
		model:              RNN that has been trained
		seq_length:         length of sequences fed to RNN as inputs
		spike_count_matrix: matrix of spike counts (rows indicate neurons; columns are time bins)
		angles:             ground truth HD angles
		times:              times (in seconds) of HD recordings
		test_stop_idx:      index for slicing test data and training data from total data

	'''
	n_times = int(angles.size - seq_length + 1)
	test_stop_idx -= seq_length
	times = times[seq_length-1:]


	model.eval()

	predict = []
	for k in range(n_times):
		cochains_tmp = torch.tensor(spike_count_matrix[..., k:k+seq_length].T, device=model.device).unsqueeze(0)
		predict.append(model(cochains_tmp.float()).detach().numpy().squeeze())



	indx = []
	# angles = (np.degrees(angles) + 180) % 360
	angles = np.degrees(angles[seq_length-1:])
	predict = np.degrees(np.array(predict))
	times = times-times[0]

	SCNNerror = np.abs(predict - angles)
	SCNNerror[SCNNerror > 180.0] = np.abs(360  - SCNNerror[SCNNerror > 180.0])
	
	SCNNcata = np.argwhere(SCNNerror > 90)

	print('RNN Catastrophic:', SCNNcata.size)


	train_MAE, train_AAE = train.MAE(predict[test_stop_idx:], angles[test_stop_idx:])
	print('train MAE:',  train_MAE)
	print('train AAE:',  train_AAE)

	test_MAE, test_AAE = train.MAE(predict[:test_stop_idx], angles[:test_stop_idx])
	print('test MAE:',  test_MAE)
	print('test AAE:',  test_AAE)

	total_MAE, total_AAE = train.MAE(predict, angles)
	print('total MAE:',  total_MAE)
	print('total AAE:',  total_AAE)

	for i in range(n_times - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times, angles, linewidth=1.5, color='blue', label='true')
		plt.plot(times, predict, linewidth=1.5, color='red', label='decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Decoded Head Angle', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/RNNmodel_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	predict_file = os.path.join(sys.path[0], "extracted/RNNmodel_prediction")
	np.savetxt(predict_file + dt + ".txt", predict, fmt = '%1.18f', delimiter=',')


	end_time = 1200
	indx = []
	for i in range(end_time - 1):
		diff = np.abs(angles[i] - angles[i+1])
		if diff > 300:
			indx.append(i)

	plt.figure(figsize=(12,5))
	if len(indx) > 0:
		plt.plot(times[:indx[0]], angles[:indx[0]], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:indx[0]], predict[:indx[0]], linewidth=1.5, color='red', label='decoded')
		for k in range(len(indx)-1):
			plt.plot(times[indx[k]+1:indx[k+1]], angles[indx[k]+1:indx[k+1]], linewidth=1.5, color='blue')
			plt.plot(times[indx[k]+1:indx[k+1]], predict[indx[k]+1:indx[k+1]], linewidth=1.5, color='red')
	else:
		plt.plot(times[:end_time], angles[:end_time], linewidth=1.5, color='blue', label='true')
		plt.plot(times[:end_time], predict[:end_time], linewidth=1.5, color='red', label='decoded')


	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Decoded Head Angle', fontsize=16)
	plt.legend()
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/RNNmodel_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	plt.figure(figsize=(12,5))

	plt.plot(times, SCNNerror, linewidth=1.5, color='red', label='SCNN')

	plt.xlabel('Time (seconds)', fontsize=14)
	plt.ylabel('Angle', fontsize=14)
	plt.title('Absolute Error Decoded Head Angle', fontsize=16)
	# plt.legend()

	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/RNNerrorplot" + dt + ".pdf"))
	# plt.show()
	plt.close()


	MAE_file = open(os.path.join(sys.path[0], "extracted/RNNtrainMAE" + dt + ".txt"), "w")
	MAE_file.write(str(train_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/RNNtrainAAE" + dt + ".txt"), "w")
	AAE_file.write(str(train_AAE))
	AAE_file.close()

	MAE_file = open(os.path.join(sys.path[0], "extracted/RNNtestMAE" + dt + ".txt"), "w")
	MAE_file.write(str(test_MAE))
	MAE_file.close()

	AAE_file = open(os.path.join(sys.path[0], "extracted/RNNtestAAE" + dt + ".txt"), "w")
	AAE_file.write(str(test_AAE))
	AAE_file.close()



