import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import time
import cv2
import torch
import torch.nn as nn
from network.MDIVDnet import MDIVDnet
from dataset import ValDataset
from val_common import denoise_seq
from utils import batch_psnr, init_logger_test, variable_to_cv2_image, close_logger, ADDNOISE_class
import numpy as np
import random
from evaluation import calc_metrics

OUTIMGEXT = '.png'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_out_seq(seqnoisy, seqclean, args, name):
	save_modeltest_dir = args['save_path']
	suffix = args['suffix']
	save_noisy = args['save_noisy']

	sigma_gaussian = int(args['val_noiseL'] * 255)
	sigma_line = int(args['val_line_noiseL'] * 255)
	sigma_row = int(args['val_row_noiseL'] * 255)
	sigma_col = int(args['val_col_noiseL'] * 255)
	sigma_bias = int(args['val_bias_noiseL'] * 255)

	save_dir = os.path.join(save_modeltest_dir, ('NB{:0>2d}_NR{:0>2d}_NC{:0>2d}_NG{:0>2d}_NL{:0>2d}'.format(sigma_bias, sigma_row, sigma_col, sigma_gaussian, sigma_line)))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	out_noise_dir = os.path.join(save_dir, name + ('noise_NB{:0>2d}_NR{:0>2d}_NC{:0>2d}_NG{:0>2d}_NL{:0>2d}').format(sigma_bias, sigma_row, sigma_col, sigma_gaussian, sigma_line))
	if len(suffix) == 0:
		out_dn_dir = os.path.join(save_dir, name + ('dn_NB{:0>2d}_NR{:0>2d}_NC{:0>2d}_NG{:0>2d}_NL{:0>2d}').format(sigma_bias, sigma_row, sigma_col, sigma_gaussian, sigma_line))
	else:
		out_dn_dir = os.path.join(save_dir, name + ('dn_NB{:0>2d}_NR{:0>2d}_NC{:0>2d}_NG{:0>2d}_NL{:0>2d}_{}').format(sigma_bias, sigma_row, sigma_col, sigma_gaussian, sigma_line, suffix))
	if save_noisy:
		if not os.path.exists(out_noise_dir):
			os.makedirs(out_noise_dir)
	if not os.path.exists(out_dn_dir):
		os.makedirs(out_dn_dir)

	seq_len = seqnoisy.size()[0]
	for idx in range(seq_len):
		# Build Outname
		fext = OUTIMGEXT
		noisy_name = os.path.join(out_noise_dir, ('{:0>4d}').format(idx) + fext)
		out_name = os.path.join(out_dn_dir, ('{:0>4d}').format(idx) + fext)

		# Save result
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			cv2.imwrite(noisy_name, noisyimg)

		outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
		cv2.imwrite(out_name, outimg)

def test_MDIVDnet(**args):
	# set seed
	seed = args['seed']
	if seed == -1:
		seed = np.random.randint(1, 10000)
	set_seed(seed)

	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])
	logger = init_logger_test(args)
	logger.info("Denoising Path: {}".format(args['test_path']))
	logger.info("model file path: {}".format(args['model_file']))

	if len(args['note']) != 0:
		logger.info(args['note'])

	print('> Loading testsets ...')
	dataset_test = ValDataset(valsetdir=args['test_path'],
							  num_input_frames=args['max_num_fr_per_seq'],
							  disp_status=False)

	if args['cuda']:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	model_temp = MDIVDnet()

	if 'net' not in args['model_file']:
		state_temp_dict_all = torch.load(args['model_file'], map_location=device)
		state_temp_dict = state_temp_dict_all['state_dict']
	else:
		state_temp_dict = torch.load(args['model_file'], map_location=device)

	if args['cuda']:
		device_ids = [0]
		model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
		checkpoint = {}
		for key, value in state_temp_dict.items():
			if key[:7] == 'module.':
				checkpoint = state_temp_dict
				break
			else:
				name = 'module.' + key
				checkpoint[name] = value
	else:
		checkpoint = {}
		for key, value in state_temp_dict.items():
			if key[:7] == 'module.':
				name = key[7:]
				checkpoint[name] = state_temp_dict
			else:
				checkpoint = state_temp_dict
				break

	model_temp.load_state_dict(checkpoint)

	model_temp.eval()

	with torch.no_grad():
		psnr_test_all = 0
		psnr_noisy_all = 0
		ssim_test_all = []
		sttred_test_all = []
		time_test_all = 0
		frame_num_count = 0
		# process data
		for seq, name in dataset_test:
			# seq = torch.from_numpy(seq).to(device)
			seq = seq.to(device)
			[F_num, C, H, W] = seq.shape
			frame_num_count += F_num

			# define noise class
			noise = ADDNOISE_class(image_size=(H, W))

			seq_time, seqn_val, out_val = denoise_seq(seq=seq,
													  noise = noise,
													  args=args,
													  model_temporal=model_temp)
			# Compute PSNR and log it
			stop_time = time.time()
			runtime = (stop_time - seq_time)

			PSNR, SSIM, STTRED = calc_metrics(out_val, seq)
			psnr_noisy = batch_psnr(seqn_val, seq, 1.)
			if args['auto_calc'] == False:
				print("scene:{}, psnr:{}, runtime:{}".format(name, PSNR, runtime))


			psnr_test_all += PSNR
			psnr_noisy_all += psnr_noisy
			time_test_all += runtime
			ssim_test_all.append(SSIM)
			sttred_test_all.append(STTRED)
			seq_length = seq.size()[0]
			logger.info("\tDenoise scene: {}, frame num: {}, runtime: {:.3f}ms".format(name, seq_length,runtime*1000))
			logger.info("\t\t\tPSNR noisy {:.4f}dB, PSNR result {:.4f}dB".format(psnr_noisy, PSNR))
			logger.info("\t\t\tSSIM {:.4f}, STTRED {:.2f},".format(SSIM, STTRED))
			
			# Save outputs
			if not args['dont_save_results']:
				save_out_seq(seqn_val, out_val, args, name)
		psnr_test_ave = psnr_test_all / len(dataset_test)
		psnr_noisy_ave = psnr_noisy_all / len(dataset_test)
		ssim_test_mean = float(np.array(ssim_test_all).mean())
		sttred_test_mean = float(np.array(sttred_test_all).mean())
		time_test_ave = time_test_all / frame_num_count * 1000
	logger.info("!!!ALL TESTSET NOISE_AVE_PSNR:{:.4f}dB, AVE_PSNR {:.4f}dB, AVE_TIME:{:.2f}ms".format(psnr_noisy_ave, psnr_test_ave, time_test_ave))
	logger.info("\t\t AVE_SSIM:{:.4f}, AVE_STTRED:{:.4f}".format(ssim_test_mean, sttred_test_mean))
	print("NOISE_AVE_PSNR:{:.2f}dB, AVE_PSNR:{:.2f}dB, AVE_TIME:{:.2f}ms".format(psnr_noisy_ave, psnr_test_ave, time_test_ave))
	print("\t\t AVE_SSIM:{:.4f}, AVE_STTRED:{:.4f}".format(ssim_test_mean, sttred_test_mean))

	# close logger
	close_logger(logger)

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
	parser.add_argument("--model_file", type=str,
						default="./pretrain/net.pth",
						help='path to model of the pretrained denoiser')
	parser.add_argument("--test_path", type=str, default="./data/test_PNG",
						help='path to sequence to denoise')
	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument('--seed', type=int, default=-1, help='random seed')
	parser.add_argument("--max_num_fr_per_seq", type=int, default=150,
						help='max number of frames to load per sequence')
	parser.add_argument('--note', type=str, default="", help='the note which is wanted to be log in logs')
	parser.add_argument("--graymode", action='store_false',
						help="whether the input image has only one channel")
	parser.add_argument("--auto_calc", action='store_false',
						help="Whether to automatically calculate 8 experimental results")

	# noise sigma
	parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on test set')
	parser.add_argument("--val_line_noiseL", type=float, default=25,
						help='noise level used on validation set')
	parser.add_argument("--val_row_noiseL", type=float, default=10,
						help='noise level used on validation set')
	parser.add_argument("--val_col_noiseL", type=float, default=10,
						help='noise level used on validation set')
	parser.add_argument("--val_bias_noiseL", type=float, default=30,
						help='noise level used on validation set')

	# noise sigma class
	parser.add_argument("--val_bias_noiseL_class", nargs=8, type=float, default=[10, 10, 10, 10, 10, 10, 10, 30],
						help='noise level used on validation set')
	parser.add_argument("--val_row_noiseL_class", nargs=8, type=float, default=[5, 5, 5, 5, 5, 15, 15, 15],
						help='noise level used on validation set')
	parser.add_argument("--val_col_noiseL_class", nargs=8, type=float, default=[5, 5, 5, 5, 15, 5, 15, 15],
						help='noise level used on validation set')
	parser.add_argument("--val_noiseL_class", nargs=8, type=float, default=[10, 10, 30, 30, 30, 30, 30, 30],
						help='noise level used on test set')
	parser.add_argument("--val_line_noiseL_class", nargs=8, type=float, default=[10, 30, 10, 30, 30, 30, 30, 30],
						help='noise level used on validation set')


	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
	parser.add_argument("--save_path", type=str, default='results',
						help='where to save outputs as png')


	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.val_bias_noiseL /= 255.
	argspar.val_row_noiseL /= 255.
	argspar.val_col_noiseL /= 255.
	argspar.val_noiseL /= 255.
	argspar.val_line_noiseL /= 255.



	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FastDVDnet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	if argspar.auto_calc == False:
		test_MDIVDnet(**vars(argspar))
	else:
		for i in range(8):
			print('noise level {}'.format(i+1))
			argspar.val_bias_noiseL = argspar.val_bias_noiseL_class[i]/255.
			argspar.val_row_noiseL = argspar.val_row_noiseL_class[i]/255.
			argspar.val_col_noiseL = argspar.val_col_noiseL_class[i]/255.
			argspar.val_noiseL = argspar.val_noiseL_class[i]/255.
			argspar.val_line_noiseL = argspar.val_line_noiseL_class[i]/255.
			test_MDIVDnet(**vars(argspar))


