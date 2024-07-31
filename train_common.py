import os
import time
import torch
from utils import batch_psnr, ADDNOISE_class
from val_common import denoise_seq
from evaluation import psnr

def get_device_and_id(gpus):
	gpus_str = gpus
	gpus = [int(gpu) for gpu in gpus.split(',')]
	gpus = [i for i in range(len(gpus))] if gpus[0] >= 0 else [-1]
	return gpus_str, gpus


def	resume_training(argdict, model, optimizer):
	""" Resumes previous training or starts anew
	"""
	if argdict['resume_training']:
		resumef = os.path.join(argdict['log_dir_resume'])
		if os.path.isfile(resumef):
			checkpoint = torch.load(resumef)
			print("> Resuming previous training")
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			new_epoch = argdict['epochs']
			new_batchsize = argdict['batch_size']
			current_lr = argdict['lr']
			argdict = checkpoint['args']
			training_params = checkpoint['training_params']
			start_epoch = training_params['start_epoch']
			argdict['epochs'] = new_epoch
			argdict['batch_size'] = new_batchsize
			argdict['lr'] = current_lr
			print("=> loaded checkpoint '{}' (epoch {})"\
				  .format(resumef, start_epoch))
			print("=> loaded parameters :")
			print("==> checkpoint['optimizer']['param_groups']")
			print("\t{}".format(checkpoint['optimizer']['param_groups']))
			print("==> checkpoint['training_params']")
			for k in checkpoint['training_params']:
				print("\t{}, {}".format(k, checkpoint['training_params'][k]))
			argpri = checkpoint['args']
			print("==> checkpoint['args']")
			for k in argpri:
				print("\t{}, {}".format(k, argpri[k]))

			argdict['resume_training'] = False
		else:
			raise Exception("Couldn't resume training with checkpoint {}".\
				   format(resumef))
	else:
		start_epoch = 0
		training_params = {}
		training_params['step'] = 0
		training_params['current_lr'] = 0

	return start_epoch, training_params

def	log_train_psnr(result, imsouce_class, loss, writer, epoch, idx, num_minibatches, training_params):
	# Log the scalar values
	psnr_train = psnr(result.detach(), imsouce_class.detach())

	print("[epoch {}][{}/{}] loss: {:1.4f} PSNR_train: {:1.4f}".\
		  format(epoch+1, idx+1, num_minibatches, loss.item(), psnr_train))

	writer.add_scalar('loss', loss.item(), training_params['step'])
	writer.add_scalars('PSNR on training val data', {'psnr_train': psnr_train}, \
					   training_params['step'])


def save_model_checkpoint(model, argdict, optimizer, train_pars, epoch, TIMESTAMP, best_info=[0, 0], psnr_val=0):
	epoch = epoch + 1
	net_dir_now = argdict['log_dir'] + '/' + TIMESTAMP[:-1] + '_netdir_' + argdict['save_all']
	if not os.path.exists(net_dir_now):
		os.makedirs(net_dir_now)
	save_dict = {
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'training_params': train_pars,
		'args': argdict
	}
	if argdict['valset_dir'] != "":
		psnr_val_up_flag = 0
		if psnr_val >= best_info[1]:
			psnr_val_up_flag = 1
			best_info[1] = psnr_val
			best_info[0] = epoch
		torch.save(model.state_dict(), os.path.join(net_dir_now, 'net.pth'))

		# save_dict
		earlierstage_epoch = int(argdict['epochs'] * 0.1)
		finalstage_epoch = int(argdict['epochs'] * 0.9)
		if argdict['save_all'] == "all":
			model_name = 'model_[{:0>2d}].pth'.format(epoch)
			torch.save(save_dict, os.path.join(net_dir_now, model_name))
		else:
			if epoch >= earlierstage_epoch & epoch < finalstage_epoch:
				if psnr_val_up_flag == 1:
					model_name = 'model_[{:0>2d}]_PSNR{:.2f}.pth'.format(epoch, best_info[1])
					torch.save(save_dict, os.path.join(net_dir_now, model_name))
			if epoch >= finalstage_epoch:
				model_name = 'model_[{:0>2d}]_bestepoch_[{:0>2d}]_PSNR{:.2f}.pth'.format(epoch, best_info[0], best_info[1])
				torch.save(save_dict, os.path.join(net_dir_now, model_name))
		if epoch == argdict['epochs']:
			model_name = 'model_last.pth'
			torch.save(save_dict, os.path.join(net_dir_now, model_name))
	else:
		torch.save(model.state_dict(), os.path.join(net_dir_now, 'net.pth'))
		psnr_val_up_flag = 0
		best_info = [0, 0]
		if argdict['save_all'] == "all":
			model_name = 'model_[{:0>2d}].pth'.format(epoch)
			torch.save(save_dict, os.path.join(net_dir_now, model_name))
	del save_dict
	return psnr_val_up_flag, best_info

def validate_and_log(model_temp, dataset_val, args,  writer, \
					 epoch, lr, training_params):
	psnr_val = 0
	with torch.no_grad():
		model_temp.eval()
		seq_index = 0
		t_total = 0
		# define noise class
		noise = ADDNOISE_class(image_size=(256, 320))

		for seq_val, name in dataset_val:
			seq_val = seq_val.cuda()
			seq_index = seq_index + 1

			t11, seqn_val, out_val = denoise_seq(seq=seq_val,
												 noise = noise,
												 args=args,
												 model_temporal=model_temp)
			t12 = time.time()
			psnr_temp = batch_psnr(out_val, seq_val, 1.)
			psnr_val += psnr_temp
			t_seq = t12 - t11
			print("    [VAL epoch %d] [%s] PSNR_val: %.4f, on %.4f sec" % (epoch + 1, name, psnr_temp, t_seq))
			t_total = t_total + t_seq
		psnr_val /= len(dataset_val)
		t_frame_ave = t_total/len(dataset_val)/args['framenum_of_val']
		print("[epoch %d] PSNR_val: %.4f, average time: %.2f ms" % (epoch+1, psnr_val, t_frame_ave*1000))
		writer.add_scalar('PSNR on validation data', psnr_val, epoch+1)
		writer.add_scalar('Learning rate', lr, epoch+1)
		writer.add_scalars('PSNR on training val data', {'psnr_val':psnr_val}, \
						  training_params['step'])

	torch.cuda.empty_cache()

	return psnr_val