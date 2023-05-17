import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os, sys
from datetime import datetime


#plot model prediction and compute perormance measurements of SCNN with seq_length=1
def plot_model_predict(model, cochains, x, y, batch_size):
	'''
	Inputs:
		model:      SCNN that has been trained
		cochains:   list of cochain tensors for each dimension
		x:          ground truth x location
		y:          ground truth y location
		batch_size: batch size used during training (output is computed on same sized batches and concatenated to get network's output across all time bins)

	'''
	n_times = x.size
	n_batches = int(n_times/batch_size)
	# cochains = [torch.tensor(C.T, device=model.device) for C in cochains]


	model.eval()

	predict_x = []
	predict_y = []
	for k in range(n_batches):
		cochains_tmp = [torch.tensor(C[..., k:k+batch_size].T, device=model.device) for C in cochains]
		pred = model(cochains_tmp).cpu().detach().numpy().squeeze()
		predict_x.append(pred[:,0])
		predict_y.append(pred[:,1])
		

	predict_x = np.concatenate(predict_x, axis=None)
	predict_y = np.concatenate(predict_y, axis=None)

	

	RMSE = np.mean(np.sqrt((x[:predict_x.size] - predict_x)**2 + (y[:predict_x.size] - predict_y)**2))
	print('Average Distance Error:', RMSE)


	plt.figure(figsize=(8,8))
	plt.plot(x, y, color='blue', label='true')
	plt.plot(predict_x, predict_y, color='red', label='decoded')


	plt.title('Decoded Trajectory', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	decoded_x_file = os.path.join(sys.path[0], "extracted/model_x_decoded")
	decoded_y_file = os.path.join(sys.path[0], "extracted/model_y_decoded")

	np.savetxt(decoded_x_file + dt + ".txt", predict_x, fmt = '%1.18f', delimiter=',')
	np.savetxt(decoded_y_file + dt + ".txt", predict_y, fmt = '%1.18f', delimiter=',')



	end_time = 600
	plt.figure(figsize=(8,8))
	plt.plot(x[:end_time], y[:end_time], color='blue', label='true')
	plt.plot(predict_x[:end_time], predict_y[:end_time], color='red', label='decoded')


	plt.title('Decodeded Trajectory', fontsize=16)
	plt.legend()
	
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	t = np.array(range(predict_x.size))*0.1


	plt.figure(figsize=(15,5))
	plt.plot(t, x[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_x, linewidth=1.4, color='red', label='decoded')

	plt.legend()
	plt.title('x-coordinate')

	plt.savefig(os.path.join("../network_scripts/plots/decoded_x" + dt + ".pdf"))

	# plt.show()
	plt.close()



	plt.figure(figsize=(15,5))
	plt.plot(t, y[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_y, linewidth=1.4, color='red', label='decoded')

	plt.legend()
	plt.title('y-coordinate')

	plt.savefig(os.path.join("../network_scripts/plots/decoded_y" + dt + ".pdf"))

	# plt.show()
	plt.close()



	plt.figure(figsize=(15,5))
	plt.plot(t, np.sqrt((x[:predict_x.size] - predict_x)**2 + (y[:predict_y.size] - predict_y)**2), linewidth=1.4, color='blue')

	plt.title('Error (Euclidean Distance)')

	plt.savefig(os.path.join("../network_scripts/plots/error" + dt + ".pdf"))

	# plt.show()
	plt.close()


#plot model prediction and compute perormance measurements of SCNN with multiple time bins considered for each input
def plot_model_predict_gen(model, seq_length, cochains, x, y, batch_size, test_stop_idx):
	'''
	Inputs:
		model:         SCNN that has been trained
		seq_length:    how many time bins are used for individual inpus into network
		cochains:      list of cochain tensors for each dimension
		x:             ground truth x location
		y:             ground truth y location
		batch_size:    batch size used during training (output is computed on same sized batches and concatenated to get network's output across all time bins)
		test_stop_idx: index for slicing test data and training data from total data

	'''
	n_times = int(x.size - seq_length + 1)
	n_batches = int(n_times/batch_size)
	
	test_stop_idx -= seq_length
	x = x[seq_length-1:]
	y = y[seq_length-1:]


	model.eval()

	predict_x = []
	predict_y = []
	for k in range(n_times):
		cochains_tmp = [torch.tensor(C[..., k:k+seq_length].T, device=model.device).unsqueeze(0) for C in cochains]
		pred = model(cochains_tmp).cpu().detach().numpy().squeeze()
		predict_x.append(pred[0])
		predict_y.append(pred[1])
		

	predict_x = np.concatenate(predict_x, axis=None)
	predict_y = np.concatenate(predict_y, axis=None)

	
	x = x[:predict_x.size]
	y = y[:predict_x.size]

	test_RMSE = np.mean(np.sqrt((x[:test_stop_idx] - predict_x[:test_stop_idx])**2 + (y[:test_stop_idx] - predict_y[:test_stop_idx])**2))
	train_RMSE = np.mean(np.sqrt((x[test_stop_idx:] - predict_x[test_stop_idx:])**2 + (y[test_stop_idx:] - predict_y[test_stop_idx:])**2))

	RMSE = np.mean(np.sqrt((x - predict_x)**2 + (y - predict_y)**2))


	print('Test Average Distance Error:', test_RMSE)
	print('Train Average Distance Error:', train_RMSE)

	print('Total Average Distance Error: ', RMSE)


	plt.figure(figsize=(8,8))
	plt.plot(x, y, color='blue', label='true')
	plt.plot(predict_x, predict_y, color='red', label='decoded')


	plt.title('Decoded Trajectory', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	decoded_x_file = os.path.join(sys.path[0], "extracted/model_x_decoded")
	decoded_y_file = os.path.join(sys.path[0], "extracted/model_y_decoded")

	np.savetxt(decoded_x_file + dt + ".txt", predict_x, fmt = '%1.18f', delimiter=',')
	np.savetxt(decoded_y_file + dt + ".txt", predict_y, fmt = '%1.18f', delimiter=',')



	end_time = 600
	plt.figure(figsize=(8,8))
	plt.plot(x[:end_time], y[:end_time], color='blue', label='true')
	plt.plot(predict_x[:end_time], predict_y[:end_time], color='red', label='decoded')


	plt.title('Decodeded Trajectory', fontsize=16)
	plt.legend()
	
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	t = np.array(range(predict_x.size))*0.1


	plt.figure(figsize=(15,5))
	plt.plot(t, x[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_x, linewidth=1.4, color='red', label='decoded')

	plt.legend()
	plt.title('x-coordinate')

	plt.savefig(os.path.join("../network_scripts/plots/decoded_x" + dt + ".pdf"))

	# plt.show()
	plt.close()



	plt.figure(figsize=(15,5))
	plt.plot(t, y[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_y, linewidth=1.4, color='red', label='decoded')

	plt.legend()
	plt.title('y-coordinate')

	plt.savefig(os.path.join("../network_scripts/plots/decoded_y" + dt + ".pdf"))

	# plt.show()
	plt.close()



	plt.figure(figsize=(15,5))
	plt.plot(t, np.sqrt((x[:predict_x.size] - predict_x)**2 + (y[:predict_y.size] - predict_y)**2), linewidth=1.4, color='blue')

	plt.title('Error (Euclidean Distance)')

	plt.savefig(os.path.join("../network_scripts/plots/error" + dt + ".pdf"))

	# plt.show()
	plt.close()


#plot model prediction and compute perormance measurements of SCRNN
def plot_model_predictSCRNN(model, seq_length, cochains, x, y, batch_size, test_stop_idx):
	'''
	Inputs:
		model:         SCRNN that has been trained
		seq_length:    how many time bins are used for individual inpus into network
		cochains:      list of cochain tensors for each dimension
		x:             ground truth x location
		y:             ground truth y location
		batch_size:    batch size used during training (output is computed on same sized batches and concatenated to get network's output across all time bins)
		test_stop_idx: index for slicing test data and training data from total data

	'''
	n_times = int(x.size - seq_length + 1)
	n_batches = int(n_times/batch_size)
	
	test_stop_idx -= seq_length
	x = x[seq_length-1:]
	y = y[seq_length-1:]


	model.eval()

	predict_x = []
	predict_y = []
	for k in range(n_times):
		cochains_tmp = [torch.tensor(C[..., k:k+seq_length].T, device=model.device).unsqueeze(0) for C in cochains]
		pred = model(cochains_tmp).cpu().detach().numpy().squeeze()
		predict_x.append(pred[0])
		predict_y.append(pred[1])
		

	predict_x = np.concatenate(predict_x, axis=None)
	predict_y = np.concatenate(predict_y, axis=None)

	
	x = x[:predict_x.size]
	y = y[:predict_x.size]

	test_RMSE = np.mean(np.sqrt((x[:test_stop_idx] - predict_x[:test_stop_idx])**2 + (y[:test_stop_idx] - predict_y[:test_stop_idx])**2))
	train_RMSE = np.mean(np.sqrt((x[test_stop_idx:] - predict_x[test_stop_idx:])**2 + (y[test_stop_idx:] - predict_y[test_stop_idx:])**2))

	RMSE = np.mean(np.sqrt((x - predict_x)**2 + (y - predict_y)**2))


	print('Test Average Distance Error:', test_RMSE)
	print('Train Average Distance Error:', train_RMSE)

	print('Total Average Distance Error: ', RMSE)


	plt.figure(figsize=(8,8))
	plt.plot(x, y, color='blue', label='true')
	plt.plot(predict_x, predict_y, color='red', label='decoded')


	plt.title('Decoded Trajectory', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	decoded_x_file = os.path.join(sys.path[0], "extracted/model_x_decoded")
	decoded_y_file = os.path.join(sys.path[0], "extracted/model_y_decoded")

	np.savetxt(decoded_x_file + dt + ".txt", predict_x, fmt = '%1.18f', delimiter=',')
	np.savetxt(decoded_y_file + dt + ".txt", predict_y, fmt = '%1.18f', delimiter=',')



	end_time = 600
	plt.figure(figsize=(8,8))
	plt.plot(x[:end_time], y[:end_time], color='blue', label='true')
	plt.plot(predict_x[:end_time], predict_y[:end_time], color='red', label='decoded')


	plt.title('Decodeded Trajectory', fontsize=16)
	plt.legend()
	
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/model_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	t = np.array(range(predict_x.size))*0.1


	plt.figure(figsize=(15,5))
	plt.plot(t, x[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_x, linewidth=1.4, color='red', label='decoded')

	plt.legend()
	plt.title('x-coordinate')

	plt.savefig(os.path.join("../network_scripts/plots/decoded_x" + dt + ".pdf"))

	# plt.show()
	plt.close()



	plt.figure(figsize=(15,5))
	plt.plot(t, y[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_y, linewidth=1.4, color='red', label='decoded')

	plt.legend()
	plt.title('y-coordinate')

	plt.savefig(os.path.join("../network_scripts/plots/decoded_y" + dt + ".pdf"))

	# plt.show()
	plt.close()



	plt.figure(figsize=(15,5))
	plt.plot(t, np.sqrt((x[:predict_x.size] - predict_x)**2 + (y[:predict_y.size] - predict_y)**2), linewidth=1.4, color='blue')

	plt.title('Error (Euclidean Distance)')

	plt.savefig(os.path.join("../network_scripts/plots/error" + dt + ".pdf"))

	# plt.show()
	plt.close()



#plot model prediction and compute perormance measurements of FFNN
def plot_model_predictFFNN(model, spike_count_matrix, x, y, test_stop_idx):
	'''
	Inputs:
		model:              FFNN that has been trained
		spike_count_matrix: matrix of spike counts (rows indicate neurons; columns are time bins)
		x:                  ground truth x location
		y:                  ground truth y location
		test_stop_idx: index for slicing test data and training data from total data

	'''
	n_times = x.size


	model.eval()

	input_tens = torch.tensor(spike_count_matrix.T)
	pred_tens = model(input_tens.float())
	predict_x = pred_tens[:,0].cpu().detach().numpy().squeeze()
	predict_y = pred_tens[:,1].cpu().detach().numpy().squeeze()

	x = x[:predict_x.size]
	y = y[:predict_x.size]

	test_RMSE = np.mean(np.sqrt((x[:test_stop_idx] - predict_x[:test_stop_idx])**2 + (y[:test_stop_idx] - predict_y[:test_stop_idx])**2))
	train_RMSE = np.mean(np.sqrt((x[test_stop_idx:] - predict_x[test_stop_idx:])**2 + (y[test_stop_idx:] - predict_y[test_stop_idx:])**2))

	RMSE = np.mean(np.sqrt((x - predict_x)**2 + (y - predict_y)**2))


	print('Test Average Distance Error:', test_RMSE)
	print('Train Average Distance Error:', train_RMSE)

	print('Total Average Distance Error: ', RMSE)


	plt.figure(figsize=(8,8))
	plt.plot(x, y, color='blue', label='true')
	plt.plot(predict_x, predict_y, color='red', label='FFNN')


	plt.title('Decoded Trajectory', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/modelFFNN_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	decoded_x_file = os.path.join(sys.path[0], "extracted/modelFFNN_x_decoded")
	decoded_y_file = os.path.join(sys.path[0], "extracted/modelFFNN_y_decoded")

	np.savetxt(decoded_x_file + dt + ".txt", predict_x, fmt = '%1.18f', delimiter=',')
	np.savetxt(decoded_y_file + dt + ".txt", predict_y, fmt = '%1.18f', delimiter=',')



	end_time = 600
	plt.figure(figsize=(8,8))
	plt.plot(x[:end_time], y[:end_time], color='blue', label='true')
	plt.plot(predict_x[:end_time], predict_y[:end_time], color='red', label='FFNN')


	plt.title('Decodeded Trajectory', fontsize=16)
	plt.legend()
	
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/modelFFNN_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	t = np.array(range(predict_x.size))*0.1


	plt.figure(figsize=(15,5))
	plt.plot(t, x[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_x, linewidth=1.4, color='red', label='FFNN')

	plt.legend()
	plt.title('x-coordinate')

	plt.margins(0,0)
	plt.savefig(os.path.join("../network_scripts/plots/FFNNdecoded_x" + dt + ".pdf"), bbox_inches = 'tight', pad_inches = 0)

	# plt.show()
	plt.close()



	plt.figure(figsize=(15,5))
	plt.plot(t, y[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_y, linewidth=1.4, color='red', label='FFNN')

	plt.legend()
	plt.title('y-coordinate')

	plt.margins(0,0)
	plt.savefig(os.path.join("../network_scripts/plots/FFNNdecoded_y" + dt + ".pdf"), bbox_inches = 'tight', pad_inches = 0)

	# plt.show()
	plt.close()



	plt.figure(figsize=(15,5))
	plt.plot(t, np.sqrt((x - predict_x)**2 + (y - predict_y)**2), linewidth=1.4, color='blue')

	plt.title('Error (Euclidean Distance)')

	plt.savefig(os.path.join("../network_scripts/plots/FFNNerror" + dt + ".pdf"))

	# plt.show()
	plt.close()





#plot model prediction and compute perormance measurements of RNN
def plot_model_predictRNN(model, seq_length, spike_count_matrix, x, y, test_stop_idx):
	'''
	Inputs:
		model:              FFNN that has been trained
		seq_length:         length of sequences fed to RNN as inputs
		spike_count_matrix: matrix of spike counts (rows indicate neurons; columns are time bins)
		x:                  ground truth x location
		y:                  ground truth y location
		test_stop_idx: index for slicing test data and training data from total data

	'''
	n_times = int(x.size - seq_length + 1)
	test_stop_idx -= seq_length
	x = x[seq_length-1:]
	y = y[seq_length-1:]


	model.eval()

	predict_x = []
	predict_y = []
	for k in range(n_times):
		cochains_tmp = torch.tensor(spike_count_matrix[..., k:k+seq_length].T, device=model.device).unsqueeze(0)
		pred = model(cochains_tmp.float()).detach().numpy().squeeze()
		predict_x.append(pred[0])
		predict_y.append(pred[1])


	predict_x = np.array(predict_x)
	predict_y = np.array(predict_y)

	test_RMSE = np.mean(np.sqrt((x[:test_stop_idx] - predict_x[:test_stop_idx])**2 + (y[:test_stop_idx] - predict_y[:test_stop_idx])**2))
	train_RMSE = np.mean(np.sqrt((x[test_stop_idx:] - predict_x[test_stop_idx:])**2 + (y[test_stop_idx:] - predict_y[test_stop_idx:])**2))

	RMSE = np.mean(np.sqrt((x - predict_x)**2 + (y - predict_y)**2))


	print('Test Average Distance Error:', test_RMSE)
	print('Train Average Distance Error:', train_RMSE)

	print('Total Average Distance Error: ', RMSE)


	plt.figure(figsize=(8,8))
	plt.plot(x, y, color='blue', label='true')
	plt.plot(predict_x, predict_y, color='red', label='decoded')


	plt.title('Decoded Trajectory', fontsize=16)
	plt.legend()
	
	now = str(datetime.now())
	dt = str(now[:10]+ '_' + str(now[11:]))

	print('time stamp: ', dt)
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/RNNmodel_approx" + dt + ".pdf"))
	# plt.show()
	plt.close()

	decoded_x_file = os.path.join(sys.path[0], "extracted/model_x_decoded")
	decoded_y_file = os.path.join(sys.path[0], "extracted/model_y_decoded")

	np.savetxt(decoded_x_file + dt + ".txt", predict_x, fmt = '%1.18f', delimiter=',')
	np.savetxt(decoded_y_file + dt + ".txt", predict_y, fmt = '%1.18f', delimiter=',')



	end_time = 600
	plt.figure(figsize=(8,8))
	plt.plot(x[:end_time], y[:end_time], color='blue', label='true')
	plt.plot(predict_x[:end_time], predict_y[:end_time], color='red', label='decoded')


	plt.title('Decodeded Trajectory', fontsize=16)
	plt.legend()
	
	
	plt.margins(0,0)
	plt.savefig(os.path.join(sys.path[0], "plots/RNNmodel_approx_first_minute" + dt + ".pdf"))
	# plt.show()
	plt.close()



	t = np.array(range(predict_x.size))*0.1


	plt.figure(figsize=(15,5))
	plt.plot(t, x[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_x, linewidth=1.4, color='red', label='decoded')

	plt.legend()
	plt.title('x-coordinate')

	plt.savefig(os.path.join("../network_scripts/plots/RNNdecoded_x" + dt + ".pdf"))

	# plt.show()
	plt.close()



	plt.figure(figsize=(15,5))
	plt.plot(t, y[:predict_x.size], linewidth=1.4, color='blue', label='true')
	plt.plot(t, predict_y, linewidth=1.4, color='red', label='decoded')

	plt.legend()
	plt.title('y-coordinate')

	plt.savefig(os.path.join("../network_scripts/plots/RNNdecoded_y" + dt + ".pdf"))

	# plt.show()
	plt.close()
