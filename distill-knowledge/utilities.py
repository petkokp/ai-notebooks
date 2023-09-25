import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def trainStep(network, criterion, optimizer, X, y):
	optimizer.zero_grad()
	outputs = network(X)
	loss = criterion(outputs, y)
	loss.backward()
	optimizer.step()
	accuracy = float(torch.sum(torch.argmax(outputs, dim=1) == y).item()) / y.shape[0]
	return loss, accuracy

def getLossAccuracyOnDataset(network, dataset_loader, fast_device, criterion=None):
	network.is_training = False
	accuracy = 0.0
	loss = 0.0
	dataset_size = 0
	for j, D in enumerate(dataset_loader, 0):
		X, y = D
		X = X.to(fast_device)
		y = y.to(fast_device)
		with torch.no_grad():
			pred = network(X)
			if criterion is not None:
				loss += criterion(pred, y) * y.shape[0]
			accuracy += torch.sum(torch.argmax(pred, dim=1) == y).item()
		dataset_size += y.shape[0]
	loss, accuracy = loss / dataset_size, accuracy / dataset_size
	network.is_training = True
	return loss, accuracy

def trainTeacherOnHparam(teacher_net, hparam, num_epochs, 
						train_loader, val_loader, 
						print_every=0, 
						fast_device=torch.device('cpu')):
	train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
	teacher_net.dropout_input = hparam['dropout_input']
	teacher_net.dropout_hidden = hparam['dropout_hidden']
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(teacher_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])
	for epoch in range(num_epochs):
		lr_scheduler.step()
		if epoch == 0:
			if val_loader is not None:
				val_loss, val_acc = getLossAccuracyOnDataset(teacher_net, val_loader, fast_device, criterion)
				val_loss_list.append(val_loss)
				val_acc_list.append(val_acc)
				print('epoch: %d validation loss: %.3f validation accuracy: %.3f' %(epoch, val_loss, val_acc))
		for i, data in enumerate(train_loader, 0):
			X, y = data
			X, y = X.to(fast_device), y.to(fast_device)
			loss, acc = trainStep(teacher_net, criterion, optimizer, X, y)
			train_loss_list.append(loss)
			train_acc_list.append(acc)
		
			if print_every > 0 and i % print_every == print_every - 1:
				print('[%d, %5d/%5d] train loss: %.3f train accuracy: %.3f' %
					  (epoch + 1, i + 1, len(train_loader), loss, acc))
		
		if val_loader is not None:
			val_loss, val_acc = getLossAccuracyOnDataset(teacher_net, val_loader, fast_device, criterion)
			val_loss_list.append(val_loss)
			val_acc_list.append(val_acc)
			print('Epoch: %d validation loss: %.3f validation accuracy: %.3f' %(epoch + 1, val_loss, val_acc))
	return {'train_loss': train_loss_list, 
			'train_acc': train_acc_list, 
			'val_loss': val_loss_list, 
			'val_acc': val_acc_list}

def studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha):
	optimizer.zero_grad()
	teacher_pred = None
	if (alpha > 0):
		with torch.no_grad():
			teacher_pred = teacher_net(X)
	student_pred = student_net(X)
	loss = studentLossFn(teacher_pred, student_pred, y, T, alpha)
	loss.backward()
	optimizer.step()
	accuracy = float(torch.sum(torch.argmax(student_pred, dim=1) == y).item()) / y.shape[0]
	return loss, accuracy

def trainStudentOnHparam(teacher_net, student_net, hparam, num_epochs, 
						train_loader, val_loader, 
						print_every=0, 
						fast_device=torch.device('cpu')):
	train_loss_list, train_acc_list, val_acc_list = [], [], []
	T = hparam['T']
	alpha = hparam['alpha']
	student_net.dropout_input = hparam['dropout_input']
	student_net.dropout_hidden = hparam['dropout_hidden']
	optimizer = optim.SGD(student_net.parameters(), lr=hparam['lr'], momentum=hparam['momentum'], weight_decay=hparam['weight_decay'])
	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=hparam['lr_decay'])

	def studentLossFn(teacher_pred, student_pred, y, T, alpha):
		if (alpha > 0):
			loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1), reduction='batchmean') * (T ** 2) * alpha + F.cross_entropy(student_pred, y) * (1 - alpha)
		else:
			loss = F.cross_entropy(student_pred, y)
		return loss

	for epoch in range(num_epochs):
		lr_scheduler.step()
		if epoch == 0:
			if val_loader is not None:
				_, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
				val_acc_list.append(val_acc)
				print('Epoch: %d validation accuracy: %.3f' %(epoch, val_acc))
		for i, data in enumerate(train_loader, 0):
			X, y = data
			X, y = X.to(fast_device), y.to(fast_device)
			loss, acc = studentTrainStep(teacher_net, student_net, studentLossFn, optimizer, X, y, T, alpha)
			train_loss_list.append(loss)
			train_acc_list.append(acc)
		
			if print_every > 0 and i % print_every == print_every - 1:
				print('[%d, %5d/%5d] train loss: %.3f train accuracy: %.3f' %
					  (epoch + 1, i + 1, len(train_loader), loss, acc))
	
		if val_loader is not None:
			_, val_acc = getLossAccuracyOnDataset(student_net, val_loader, fast_device)
			val_acc_list.append(val_acc)
			print('Epoch: %d validation accuracy: %.3f' %(epoch + 1, val_acc))
	return {'train_loss': train_loss_list, 
			'train_acc': train_acc_list, 
			'val_acc': val_acc_list}

def hparamToString(hparam):
	hparam_str = ''
	for k, v in sorted(hparam.items()):
		hparam_str += k + '=' + str(v) + ', '
	return hparam_str[:-2]

def hparamDictToTuple(hparam):
	hparam_tuple = [v for k, v in sorted(hparam.items())]
	return tuple(hparam_tuple)

def getTrainMetricPerEpoch(train_metric, updates_per_epoch):
	train_metric_per_epoch = []
	temp_sum = 0.0
	for i in range(len(train_metric)):
		temp_sum += train_metric[i]
		if (i % updates_per_epoch == updates_per_epoch - 1):
			train_metric_per_epoch.append(temp_sum / updates_per_epoch)
			temp_sum = 0.0

	return train_metric_per_epoch