import os
import pandas as pd
from tqdm.notebook import tqdm
from util.TranAD.models import *
from util.TranAD.constants import *
from util.TranAD.pot import *
from util.TranAD.utils import *
from util.TranAD.diagnosis import *
from util.TranAD.merlin import *
from util.util_overlap import find_length
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time

import matplotlib.pyplot as plt
import os, torch
import numpy as np
import random


def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
	return torch.stack(windows)

def load_dataset(dataset, d_type, cd1, cd2, character, val, width):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	# print(loader)

	test_d = loader[1].astype(float)
	test_d = test_d.reshape(len(test_d),)
	label_d = loader[2].reshape(len(loader[2]),)

	win = find_length(test_d)

	test_d, label_d, label_cd = add_drift(test_d, label_d, d_type, cd1, cd2, character, val, win, width)

	loader[1] = test_d.reshape(len(test_d),1)
	loader[2] = label_d.reshape(len(label_d),1)
    
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels, label_cd, loader 


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import util.TranAD.models
	model_class = getattr(util.TranAD.models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = dataO.shape[1]
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data: 
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, None)
				else:
					x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'MAD_GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1; w_size = model.n_window
		mses, gls, dls = [], [], []
		if training:
			for d in data:
				# training discriminator
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# training generator
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
				# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
		bs = model.batch if training else len(data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d, _ in dataloader:
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			# tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			print(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			for d, _ in dataloader:
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
			loss = l(z, elem)[0]
			return loss.detach().numpy(), z.detach().numpy()[0]
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels, threshold):
	if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	# comments out (just show in the cell) // jj
	# os.makedirs(os.path.join('plots', name), exist_ok=True)
	# pdf = PdfPages(f'plots/{name}/output.pdf')
	for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		ax2.plot(smooth(a_s), linewidth=0.2, color='g')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		ax2.axhline(y=threshold, color='r', linewidth=1)
		# pdf.savefig(fig)	# // jj
		# plt.close()
	# pdf.close()

def training_model(train_loader, test_loader, NN_model, labels, dataset):
    if NN_model in ['MERLIN']:
        eval(f'run_{NN_model.lower()}(test_loader, labels, dataset)')
    
    model, optimizer, scheduler, epoch, accuracy_list = load_model(NN_model, labels.shape[1])
    
    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

    print(f'{color.HEADER}Training {NN_model} on {dataset}{color.ENDC}')
    num_epochs = 5; e = epoch + 1; start = time()
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
        lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
        accuracy_list.append((lossT, lr))
    print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
    save_model(model, optimizer, scheduler, e, accuracy_list)
	# plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
	# plot_accuracies(accuracy_list)
    return model, testD, testO, optimizer, scheduler

def plot_accuracies(accuracy_list):
	# os.makedirs(f'plots/{folder}/', exist_ok=True)
	trainAcc = [i[0] for i in accuracy_list]
	lrs = [i[1] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.twinx()
	plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
	# plt.savefig(f'plots/{folder}/training-graph.pdf')
	# plt.clf()

# To use the univariate dataset from TSB-UAD // jj
def gradPattern(p1, p2, width):
	l = len(p1)
	num = round(width / l)
	seq = np.array([])
	for i in range(num):
		weight = [num-i, i+1]
		select = p1 if random.choices(range(0,2), weights = weight)[0] ==0 else p2
		seq = np.concatenate((seq, select))
	seq = np.concatenate((seq, p2[:width-num*l]))
	return seq


# type is a list
from scipy import signal
def add_drift(data, label, d_type, r_cd1, r_cd2, character, val, win, width=None):
    label_cd = np.zeros(len(label))
    
    if d_type =='None':
        return data, label, label_cd
    cd1 = round(len(data)*r_cd1)
    cd2 = round(len(data)*r_cd2)
    label_cd[cd1:cd2] = np.ones(cd2-cd1)
                

    if not width==None:
        width = round((cd2-cd1)*width)
    if d_type == 'abrupt':
        if character == 'mean':
            data2 = np.concatenate((data[:cd1], data[cd1:cd2] + val, data[cd2:]))
            label2 = label
        elif character == 'mean_flip':
            data2 = np.concatenate((data[:cd1], np.flip(data[cd1:cd2]) + val, data[cd2:]))
            label2 = np.concatenate((label[:cd1], np.flip(label[cd1:cd2]), label[cd2:]))
        elif character == 'freq':
            # in here, val = ratio (if ratio < 1 -> inc. freq.)
            d_temp = data[cd1:cd2]
            wid_len = int((cd2-cd1)*val)
            d_mod = signal.resample(d_temp, wid_len)
            l_temp = label[cd1:cd2]            
            l_mod = signal.resample(l_temp, wid_len)
            l_mod = np.round(l_mod)
            data2 = np.concatenate((data[:cd1], d_mod, data[cd2:])) # caution! the length of data is changed here
            # data2 = np.concatenate((data[:cd1], np.flip(d_mod), data[cd2:])) # caution! the length of data is changed here
            label2 = np.concatenate((label[:cd1], l_mod, label[cd2:])) # caution! the length of data is changed here
            # label2 = np.concatenate((label[:cd1], np.flip(l_mod), label[cd2:])) # caution! the length of data is changed here
            label_cd = np.zeros(len(label2))
            label_cd[cd1:cd1+len(l_mod)] = np.ones(len(l_mod))
        elif character == 'freq_flip':
    # in here, val = ratio (if ratio < 1 -> inc. freq.)
            d_temp = data[cd1:cd2]
            wid_len = int((cd2-cd1)*val)
            d_mod = signal.resample(d_temp, wid_len)
            l_temp = label[cd1:cd2]            
            l_mod = signal.resample(l_temp, wid_len)
            l_mod = np.round(l_mod)
            # data2 = np.concatenate((data[:cd1], d_mod, data[cd2:])) # caution! the length of data is changed here
            data2 = np.concatenate((data[:cd1], np.flip(d_mod), data[cd2:])) # caution! the length of data is changed here
            # label2 = np.concatenate((label[:cd1], l_mod, label[cd2:])) # caution! the length of data is changed here
            label2 = np.concatenate((label[:cd1], np.flip(l_mod), label[cd2:])) # caution! the length of data is changed here     
            label_cd = np.zeros(len(label2))
            label_cd[cd1:cd1+len(l_mod)] = np.ones(len(l_mod))
    elif d_type == 'inc':        
        if character == 'mean':
            print(width, val)
            add1 = np.arange(width)*val/width    
            add2 = add1[::-1]
            # data2 = np.concatenate((data[:cd1], data[cd1:cd1+width]+add1, data[cd1+width:cd2]+val,data[cd2:cd2+width]+add2, data[cd2+width:] ))
            data2 = np.concatenate((data[:cd1], data[cd1:cd1+width]+add1, data[cd1+width:cd2]+val,data[cd2:cd2+width]+add2, data[cd2+width:] ))
            label2 = label            
            label_cd[cd2:cd2+width] = np.ones(width)
        elif character == 'freq':
            ratio = val
            d_temp1 = data[cd1:cd1+width]
            wid_len = int(width*(1+ratio)/2)
            mod1 = signal.resample(d_temp1, wid_len)
            l_temp1 = label[cd1:cd1+width]
            l1 = signal.resample(l_temp1, wid_len)
            l1 = np.round(l1)

            d_temp2 = data[cd1+width:cd2]
            wid_len = int((cd2-cd1-width)*ratio)
            mod2 = signal.resample(d_temp2, wid_len)
            l_temp2 = label[cd1+width:cd2]
            l2 = signal.resample(l_temp2, wid_len)
            l2 = np.round(l2)

            d_temp3 = data[cd2:cd2+width]
            wid_len = int(width*(1+ratio)/2)
            mod3 = signal.resample(d_temp3, wid_len)
            l_temp3 = label[cd2:cd2+width]
            l3 = signal.resample(l_temp3, wid_len)
            l3 = np.round(l3)

            # print('Compare:', len(mod1), len(mod2), len(mod3))

            data2 = np.concatenate((data[:cd1], mod1, mod2, mod3, data[cd2+width:] ))
            label2 = np.concatenate((label[:cd1], l1, l2, l3, label[cd2+width:] ))
            label_cd = np.zeros(len(label2))
            len_cd = len(l1) + len(l2) + len(l3)
            label_cd[cd1:cd1+len_cd] = np.ones(len_cd)
	    
    elif d_type == 'grad':        
        if character == 'mean_flip':
            s_ind = 0
            p1 = data[s_ind:s_ind+win]
            # print(s_ind, len(p1), np.count_nonzero(p1))
            while np.count_nonzero(label[s_ind:s_ind+win]) >0:
             s_ind = s_ind+win
             p1 = data[s_ind:s_ind+win]
            #  print(s_ind, len(p1),np.count_nonzero(p1))
            p2 = np.flip(p1) + val               
            w1 = gradPattern(p1, p2, width)            
            w2 = gradPattern(p2, p1, width)
            # plt.plot(w1)
            # plt.plot(w2)
            # plt.show()
            data2 = np.concatenate((data[:cd1], w1, np.flip(data[cd1+width:cd2]) +val,  w2, data[cd2+width:]))
            label2 = np.concatenate((label[:cd1], np.zeros(len(w1)), np.flip(label[cd1+width:cd2]), np.zeros(len(w2)),label[cd2+width:]))

    return data2, label2, label_cd
            

# file open 
def preprocess_univariate(dataset, max_length, train=None):
	if dataset == 'Dodgers':  file = '101-freeway-traffic.test.out'
	if 'ECG' in dataset: file = 'MBA_ECG805_data.out'
	if dataset == 'ECG14046': file = 'MBA_ECG14046_data.out'
	if dataset == 'KDD21': file = '001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.out'
	if dataset == 'MGAB': file = '1.test.out'
	if dataset == 'NAB': file = 'NAB_data_art0_0.out'
	if dataset == 'SensorScope': file = 'stb-3.test.out'
	if dataset == 'YAHOO': file = 'YahooA3Benchmark-TS1_data.out'
	if dataset == 'IOPS' :
		file_train = 'KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa.train.out'
		file = 'KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa.test.out'
	if dataset == 'NASA-MSL': 
		file_train = 'C-1.train.out'
		file = 'C-1.test.out'
	if dataset == 'NASA-SMAP':
		file_train = 'A-1.train.out'
		file = 'A-1.test.out'
	# if dataset == 'artificial':
    	# file = './data/artificial/0_2_0.01_5.out'
	filepath = './data/benchmark/' + dataset + '/' + file
	if dataset == 'ECG14046':
		filepath = './data/benchmark/ECG/' + file
	if dataset == 'ECG_1':
		filepath = './data/synthetic/ECG_add_white_noise_0.5/' + file
	if dataset == 'ECG_2':
		filepath = './data/synthetic/ECG_change_segment_add_scale_0.08/' + file
	if dataset == 'ECG_3':
		filepath = './data/synthetic/ECG_change_segment_resampling_0.08/' + file
	if dataset == 'ECG_4':
		filepath = './data/synthetic/ECG_filter_fft_21/' + file
	if dataset == 'ECG_5':
		filepath = './data/synthetic/ECG_flip_segment_0.08/' + file
	if dataset == 'ECG_6':
		filepath = './data/synthetic/ECG_flat_region_0.04/' + file
	# filepath = './data/benchmark/ECG/MBA_ECG805_data.out'
	# if dataset == 'artificial':
    	# filepath = './data/artificial/5_2_0.02_11.out'

	# uni_data = ['Dodgers', 'ECG', 'KDD21', 'MGAB', 'NAB', 'SensorScope', 'YAHOO']
	# if any(k in dataset for k in uni_data):

	if dataset in ['IOPS', 'NASA-MSL', 'NASA-SMAP']:
		if train == 'train':
			filepath = './data/benchmark/' + dataset + '/' + file_train

	data_org = pd.read_csv(filepath, header=None).to_numpy()
	name = filepath.split('/')[-1]

	# for reducing size of dataset
	if len(data_org) < max_length or max_length <=0:
		max_length = len(data_org)

	if dataset == 'Dodgers':
		data_org = data_org / 10
		s_len = 400
		if max_length < len(data_org):
			max_length = max_length+s_len
		data = data_org[s_len:max_length,0].astype(float)
		label = data_org[s_len:max_length,1]
	elif dataset == 'YAHOO':
		# data_org = data_org + 5000
		data = data_org[:max_length,0].astype(float)
		label = data_org[:max_length,1]
	else:
		data = data_org[:max_length,0].astype(float)
		label = data_org[:max_length,1]

	return data, label, name 

def load_Uni_dataset(dataset, d_type, cd1, cd2, character, val, width):
    # folder = os.path.join('./data/benchmark/', dataset)
	loader = []

    # for Unsupervised 
	uni_data = ['Dodgers', 'ECG', 'KDD21', 'MGAB', 'NAB', 'SensorScope', 'YAHOO']
	if any(k in dataset for k in uni_data):
        
		data, label, name = preprocess_univariate(dataset, max_length=30000)

		# impute simple incremental conecpt drift here		
		win = find_length(data)
		data, label, label_cd = add_drift(data, label, d_type, cd1, cd2, character, val, win, width)
		# prepare data for training and testing. Here, the training ratio = 0.2
		loader.append(data[:int(0.2*len(data))].reshape(int(0.2*len(data)),1))
		loader.append(data.reshape(len(data),1))
		loader.append(label.reshape(len(label),1))
        
	elif dataset in ['IOPS', 'NASA-MSL', 'NASA-SMAP']:
		data_tr, label_tr, name_tr = preprocess_univariate(dataset, max_length=-1, train='train')
		data_t, label_t, name = preprocess_univariate(dataset, max_length=-1)

		data_t, label_t, label_cd = add_drift(data, label, d_type, cd1, cd2, character, val, win, width)

		label = np.concatenate(label_tr, label_t)
		
		loader.append(data_tr.reshape(len(data_tr),1))
		loader.append(data_t.reshape(len(data_t),1))
		loader.append(label_t.reshape(len(label_t),1))
        

	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
    
	return train_loader, test_loader, labels, label_cd, loader, name
