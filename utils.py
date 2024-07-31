import os
import subprocess
import glob
import logging
from random import choices # requires Python >= 3.6
import numpy as np
import cv2
import torch
# from skimage.measure.simple_metrics import compare_psnr #for skimage 0.16
from skimage.metrics import peak_signal_noise_ratio as compare_psnr #for skimage 0.24
from tensorboardX import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
import random
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import scipy.ndimage
import scipy.linalg


IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types

def normalize_augment(datain):
	def transform(sample):
		do_nothing = lambda x: x
		do_nothing.__name__ = 'do_nothing'
		flipud = lambda x: torch.flip(x, dims=[2])
		flipud.__name__ = 'flipup'
		rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
		rot90.__name__ = 'rot90'
		rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
		rot90_flipud.__name__ = 'rot90_flipud'
		rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
		rot180.__name__ = 'rot180'
		rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
		rot180_flipud.__name__ = 'rot180_flipud'
		rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
		rot270.__name__ = 'rot270'
		rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
		rot270_flipud.__name__ = 'rot270_flipud'
		add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
											  std=(5/255.)).expand_as(x).to(x.device)
		add_csnt.__name__ = 'add_csnt'

		aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
					rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
		w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12]
		transf = choices(aug_list, w_aug)

		return transf[0](sample)

	img_train = datain
	img_train = img_train.view(img_train.size()[0], -1,
							   img_train.size()[-2], img_train.size()[-1]) / 255.

	img_train = transform(img_train)

	return img_train

def init_logging(argdict):
	TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
	if argdict['noteindir'] == "":
		log_dir_now = argdict['log_dir'] + '/' + TIMESTAMP
	else:
		log_dir_now = argdict['log_dir'] + '/' + TIMESTAMP[:-1] + '___' + argdict['noteindir'] + '/'
	if not os.path.exists(log_dir_now):
		os.makedirs(log_dir_now)
	writer = SummaryWriter(log_dir_now)
	logger = init_logger(log_dir_now, argdict)
	logger.info("result_DIR: {}".format(TIMESTAMP))
	print("result_DIR: {}".format(TIMESTAMP))

	return writer, logger, TIMESTAMP

def get_imagenames(seq_dir, pattern=None):
	files = []
	for typ in IMAGETYPES:
		files.extend(glob.glob(os.path.join(seq_dir, typ)))
	if not pattern is None:
		ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
		files = ffiltered
		del ffiltered

	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return files

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100, disp_status=True):
	files = get_imagenames(seq_dir)

	files_num = files.__len__()
	if files_num < max_num_fr:
		read_frame_num = files_num
	else:
		read_frame_num = max_num_fr
	seq_list = []
	if disp_status == True:
		print("\tOpen sequence in folder: ", seq_dir)
	for fpath in files[0:read_frame_num]:

		img, expanded_h, expanded_w = open_image(fpath,
												 gray_mode=gray_mode,
												 expand_if_needed=expand_if_needed,
												 expand_axis0=False)
		seq_list.append(img)
	seq = np.stack(seq_list, axis=0)
	return seq, expanded_h, expanded_w

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
	if not gray_mode:
		img = cv2.imread(fpath)
		img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

	if expand_axis0:
		img = np.expand_dims(img, 0)

	expanded_h = False
	expanded_w = False
	sh_im = img.shape
	if expand_if_needed:
		if sh_im[-2]%2 == 1:
			expanded_h = True
			if expand_axis0:
				img = np.concatenate((img, \
									  img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
			else:
				img = np.concatenate((img, \
									  img[:, -1, :][:, np.newaxis, :]), axis=1)


		if sh_im[-1]%2 == 1:
			expanded_w = True
			if expand_axis0:
				img = np.concatenate((img, \
									  img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
			else:
				img = np.concatenate((img, \
									  img[:, :, -1][:, :, np.newaxis]), axis=2)

	if normalize_data:
		img = normalize(img)
	return img, expanded_h, expanded_w

def batch_psnr(img, imclean, data_range):
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
							 data_range=data_range)
	return psnr/img_cpu.shape[0]

def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
	assert torch.max(invar) <= 1.0

	size4 = len(invar.size()) == 4
	if size4:
		nchannels = invar.size()[1]
	else:
		nchannels = invar.size()[0]

	if nchannels == 1:
		if size4:
			res = invar.data.cpu().numpy()[0, 0, :]
		else:
			res = invar.data.cpu().numpy()[0, :]
		res = (res*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		if size4:
			res = invar.data.cpu().numpy()[0]
		else:
			res = invar.data.cpu().numpy()
		res = res.transpose(1, 2, 0)
		res = (res*255.).clip(0, 255).astype(np.uint8)
		if conv_rgb_to_bgr:
			res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
	else:
		raise Exception('Number of color channels not supported')
	return res

def get_git_revision_short_hash():
	r"""Returns the current Git commit.
	"""
	return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(log_dir, argdict):
	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(log_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	if len(argdict['note']) != 0:
		logger.info(argdict['note'])

	try:
		logger.info("Commit: {}".format(get_git_revision_short_hash()))
	except Exception as e:
		logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.keys():
		logger.info("\t{}: {}".format(k, argdict[k]))

	return logger

def init_logger_test(argdicts):
	from os.path import join
	result_dir = argdicts['save_path']
	noise_gaussian_sigma = int(argdicts['val_noiseL'] *255)
	noise_line_sigma = int(argdicts['val_line_noiseL'] *255)
	noise_row_sigma = int(argdicts['val_row_noiseL'] * 255)
	noise_col_sigma = int(argdicts['val_col_noiseL'] * 255)
	noise_bias_sigma = int(argdicts['val_bias_noiseL'] * 255)

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log'+'NB'+'{:0>2d}'.format(noise_bias_sigma)+'_NR'+'{:0>2d}'.format(noise_row_sigma)+'_NC'+'{:0>2d}'.format(noise_col_sigma)+'_NG'+'{:0>2d}'.format(noise_gaussian_sigma)+'_NL'+'{:0>2d}'.format(noise_line_sigma)+'.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def close_logger(logger):
	x = list(logger.handlers)
	for i in x:
		logger.removeHandler(i)
		i.flush()
		i.close()

def normalize(data):
	data = np.float32(data)
	data = (data-data.min())/(data.max()-data.min())
	return data

def remove_dataparallel_wrapper(state_dict):
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict

def genBiasField_ori(SIZE):
	size_center = min(SIZE)//2
	W_pos = size_center
	num_points = 4
	cx = np.linspace(W_pos, W_pos, num=num_points)
	H_pos = size_center
	cy = np.geomspace(H_pos, H_pos, num=num_points)
	coils = np.stack([cy, cx], axis=0).T
	cy_out = np.array([H_pos, -3 * H_pos, 5 * H_pos, H_pos])
	cx_out = np.array([-3 * W_pos, W_pos, W_pos, 5 * W_pos])
	coils_out = np.stack([cy_out, cx_out], axis=0).T
	B = np.zeros((min(SIZE), min(SIZE)))
	for i in range(B.shape[0]):
		for j in range(B.shape[1]):
			p = np.array([i, j])
			p = np.tile(p, (num_points, 1))
			dist1 = np.min(np.linalg.norm(coils - p, axis=1))
			dist2 = np.min(np.linalg.norm(coils_out - p, axis=1))
			if dist1 <= 10:
				dist1 = 10
			B[i, j] = -np.log(dist2/dist1)
	B_norm = normalize(B)
	B_norm = pow(B_norm, 4)
	B_norm = cv2.resize(B_norm, (SIZE[1], SIZE[0]))
	return B_norm

class ADDNOISE_class():
	def __init__(self, image_size):
		super(ADDNOISE_class, self).__init__()
		self.bias_field_ori = genBiasField_ori(image_size)
		self.SIZE = image_size

	def addnoise(self, x, args, mode='train'):
		graymode = args['graymode']
		device = x.device
		t = x
		if graymode == True:
			C = 1
		else:
			C = 3
		if mode == 'train':
			N, FC, H, W = x.shape
			F = int(FC / C)
			t = t.reshape(N, F, C, H, W)
			stdn = torch.empty((N, 1, 1, 1, 1)).to(device).uniform_(args['noise_ival'][0], to=args['noise_ival'][1])
			stdnline = torch.empty((N, 1, 1, 1, 1)).to(device).uniform_(args['noise_line_ival'][0],
																	to=args['noise_line_ival'][1])
			stdnrow_spatial = torch.empty((N, 1, 1, 1, 1)).to(device).uniform_(args['noise_rowspatial_ival'][0],
																		   to=args['noise_rowspatial_ival'][1])
			stdncol_spatial = torch.empty((N, 1, 1, 1, 1)).to(device).uniform_(args['noise_colspatial_ival'][0],
																		   to=args['noise_colspatial_ival'][1])
			temperature_Coefficient = torch.empty((N, 1, 1, 1, 1)).to(device).uniform_(args['noise_bias_ival'][0],
																				   to=args['noise_bias_ival'][1])
			patchsize = args['patch_size']

		if mode == 'val' or mode == 'test':
			t = t.unsqueeze(dim=0)
			N, F, C, H, W = t.shape
			stdn = torch.FloatTensor([args['val_noiseL']]).repeat(N, 1, 1, 1, 1).to(device)
			stdnline = torch.FloatTensor([args['val_line_noiseL']]).repeat(N, 1, 1, 1, 1).to(device)
			stdnrow_spatial = torch.FloatTensor([args['val_row_noiseL']]).repeat(N, 1, 1, 1, 1).to(device)
			stdncol_spatial = torch.FloatTensor([args['val_col_noiseL']]).repeat(N, 1, 1, 1, 1).to(device)
			temperature_Coefficient = torch.FloatTensor([args['val_bias_noiseL']]).repeat(N, 1, 1, 1, 1).to(device)
			patchsize = None
		noise_gaussian = torch.zeros(N, F, 1, H, W).to(device)
		noise_gaussian = torch.normal(mean=noise_gaussian, std=stdn.expand_as(noise_gaussian))
		noise_gaussian = noise_gaussian.repeat(1, 1, C, 1, 1)
		line_noise_time = torch.zeros([N, F, 1, H, 1]).to(device)
		line_noise_time = torch.normal(mean=line_noise_time, std=stdnline.expand_as(line_noise_time))
		line_noise_time = line_noise_time.repeat(1, 1, C, 1, W)
		row_noise_spatial = torch.zeros([N, 1, 1, H, 1]).to(device)
		row_noise_spatial = torch.normal(mean=row_noise_spatial, std=stdnrow_spatial.expand_as(row_noise_spatial))
		row_noise_spatial = row_noise_spatial.repeat(1, F, C, 1, W)
		col_noise_spatial = torch.zeros([N, 1, 1, 1, W]).to(device)
		col_noise_spatial = torch.normal(mean=col_noise_spatial, std=stdncol_spatial.expand_as(col_noise_spatial))
		col_noise_spatial = col_noise_spatial.repeat(1, F, C, H, 1)
		noise_bias_field = self.add_biasfield(patchsize=patchsize, intensity=temperature_Coefficient)
		noise_bias_field = noise_bias_field.repeat(1, F, C, 1, 1)
		noise_gaussian = noise_gaussian.to(device)
		line_noise_time = line_noise_time.to(device)
		row_noise_spatial = row_noise_spatial.to(device)
		col_noise_spatial = col_noise_spatial.to(device)
		noise_bias_field = noise_bias_field.to(device)
		imgn = t + row_noise_spatial + col_noise_spatial + noise_bias_field + noise_gaussian + line_noise_time
		imgn = imgn.clamp(0., 1.)

		if mode == 'train':
			imgn = imgn.reshape(N, FC, H, W).to(device)
		else:
			imgn = imgn.reshape(F, C, H, W).to(device)
		return imgn

	def add_biasfield(self, patchsize=96, intensity=1):
		device = intensity.device
		N = intensity.shape[0]
		if patchsize != None:
			H = patchsize
			W = patchsize
		else:
			H = self.SIZE[0]
			W = self.SIZE[1]
		Bias_field = torch.zeros(N, 1, 1, H, W)

		for i in range(N):
			B_boosted = Image.fromarray(self.bias_field_ori)
			if patchsize != None:
				nw = random.randint(0, B_boosted.size[1] - patchsize)  ##裁剪图像在原图像中的坐标
				nh = random.randint(0, B_boosted.size[0] - patchsize)
				B_boosted = B_boosted.crop((nh, nw, nh + patchsize, nw + patchsize))

			to_tensor = transforms.ToTensor()
			B_boosted = to_tensor(B_boosted).unsqueeze(dim=0).unsqueeze(dim=0)
			Bias_field[i, 0, 0, :, :] = B_boosted

		Bias_field = Bias_field.to(device)
		Bias_field = Bias_field * intensity - intensity / 2

		return Bias_field


def flow_warp(x,
			  flow,
			  interpolation='bilinear',
			  padding_mode='zeros',
			  align_corners=True):

	if x.size()[-2:] != flow.size()[1:3]:
		raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
						 f'flow ({flow.size()[1:3]}) are not the same.')
	_, _, h, w = x.size()
	# create mesh grid
	grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w)) #torch.Size([h, w])
	grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
	grid.requires_grad = False

	grid_flow = grid + flow
	# scale grid_flow to [-1,1]
	grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
	grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
	grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
	output = F.grid_sample(
		x,
		grid_flow,
		mode=interpolation,
		padding_mode=padding_mode,
		align_corners=align_corners)
	return output

class LayerNormFunction(torch.autograd.Function):

	def forward(ctx, x, weight, bias, eps):
		ctx.eps = eps
		N, C, H, W = x.size()
		mu = x.mean(1, keepdim=True)
		var = (x - mu).pow(2).mean(1, keepdim=True)
		y = (x - mu) / (var + eps).sqrt()
		ctx.save_for_backward(y, var, weight)
		y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
		return y

	def backward(ctx, grad_output):
		eps = ctx.eps

		N, C, H, W = grad_output.size()
		y, var, weight = ctx.saved_variables
		g = grad_output * weight.view(1, C, 1, 1)
		mean_g = g.mean(dim=1, keepdim=True)

		mean_gy = (g * y).mean(dim=1, keepdim=True)
		gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
		return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
			dim=0), None

class LayerNorm2d(nn.Module):

	def __init__(self, channels, eps=1e-6):
		super(LayerNorm2d, self).__init__()
		self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
		self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
		self.eps = eps

	def forward(self, x):
		return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

def loss_calc(args,
			  criterion_L2,
			  img_train,
			  out_train):
	# Compute loss of result
	N, FC, H, W = img_train.shape
	if args['graymode'] == True:
		C = 1
	else:
		C = 3
	F = FC//C

	loss_L2 = criterion_L2(img_train, out_train) / F

	loss = loss_L2

	return loss




def est_params(frame, blk, sigma_nn):
    h, w = frame.shape
    sizeim = np.floor(np.array(frame.shape)/blk) * blk
    sizeim = sizeim.astype(int)

    frame = frame[:sizeim[0], :sizeim[1]]

    #paired_products
    temp = []
    for u in range(blk):
      for v in range(blk):
        temp.append(np.ravel(frame[v:(sizeim[0]-(blk-v)+1), u:(sizeim[1]-(blk-u)+1)]))
    temp = np.array(temp).astype(np.float32)

    cov_mat = np.cov(temp, bias=1).astype(np.float32)

    # force PSD
    eigval, eigvec = np.linalg.eig(cov_mat)
    Q = np.matrix(eigvec)
    xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
    cov_mat = Q*xdiag*Q.T

    temp = []
    for u in range(blk):
      for v in range(blk):
        temp.append(np.ravel(frame[v::blk, u::blk]))
    temp = np.array(temp).astype(np.float32)

    # float32 vs float64 difference between python2 and python3
    # avoiding this problem with quick cast to float64
    V,d = scipy.linalg.eigh(cov_mat.astype(np.float64))
    V = V.astype(np.float32)

    # Estimate local variance
    sizeim_reduced = (sizeim/blk).astype(int)
    ss = np.zeros((sizeim_reduced[0], sizeim_reduced[1]), dtype=np.float32)
    if np.max(V) > 0:
      # avoid the matrix inverse for extra speed/accuracy
      ss = scipy.linalg.solve(cov_mat, temp)
      ss = np.sum(np.multiply(ss, temp) / (blk**2), axis=0)
      ss = ss.reshape(sizeim_reduced)

    V = V[V>0]

    # Compute entropy
    ent = np.zeros_like(ss, dtype=np.float32)
    for u in range(V.shape[0]):
      ent += np.log2(ss * V[u] + sigma_nn) + np.log(2*np.pi*np.exp(1))


    return ss, ent


def extract_info(frame1, frame2):
    blk = 3
    sigma_nsq = 0.1
    sigma_nsqt = 0.1

    model = SpatialSteerablePyramid(height=6)
    y1 = model.extractSingleBand(frame1, filtfile="sp5Filters", band=0, level=4)
    y2 = model.extractSingleBand(frame2, filtfile="sp5Filters", band=0, level=4)

    ydiff = y1 - y2

    ss, q = est_params(y1, blk, sigma_nsq)
    ssdiff, qdiff = est_params(ydiff, blk, sigma_nsqt)


    spatial = np.multiply(q, np.log2(1 + ss))
    temporal = np.multiply(qdiff, np.multiply(np.log2(1 + ss), np.log2(1 + ssdiff)))

    return spatial, temporal



def strred(referenceVideoData, distortedVideoData):
    """Computes Spatio-Temporal Reduced Reference Entropic Differencing (ST-RRED) Index. [#f1]_

    Both video inputs are compared over frame differences, with quality determined by
    differences in the entropy per subband.

    Parameters
    ----------
    referenceVideoData : ndarray
        Reference video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    distortedVideoData : ndarray
        Distorted video, ndarray of dimension (T, M, N, C), (T, M, N), (M, N, C), or (M, N),
        where T is the number of frames, M is the height, N is width,
        and C is number of channels. Here C is only allowed to be 1.

    Returns
    -------
    strred_array : ndarray
        The ST-RRED results, ndarray of dimension ((T-1)/2, 4), where T
        is the number of frames.  Each row holds spatial score, temporal score,
        reduced reference spatial score, and reduced reference temporal score.

    strred : float
        The final ST-RRED score if all blocks are averaged after comparing
        reference and distorted data. This is close to full-reference.

    strredssn : float
        The final ST-RRED score if all blocks are averaged before comparing
        reference and distorted data. This is the reduced reference score.

    References
    ----------

    .. [#f1] R. Soundararajan and A. C. Bovik, "Video Quality Assessment by Reduced Reference Spatio-temporal Entropic Differencing," IEEE Transactions on Circuits and Systems for Video Technology, April 2013.

    """

    referenceVideoData = vshape(referenceVideoData)
    distortedVideoData = vshape(distortedVideoData)

    assert(referenceVideoData.shape == distortedVideoData.shape)

    T, M, N, C = referenceVideoData.shape

    assert C == 1, "strred called with videos containing %d channels. Please supply only the luminance channel" % (C,)

    referenceVideoData = referenceVideoData[:, :, :, 0]
    distortedVideoData = distortedVideoData[:, :, :, 0]

    rreds = []
    rredt = []

    rredssn = []
    rredtsn = []

    for i in range(0, T-1, 2):
      refFrame1 = referenceVideoData[i].astype(np.float32)
      refFrame2 = referenceVideoData[i+1].astype(np.float32)

      disFrame1 = distortedVideoData[i].astype(np.float32)
      disFrame2 = distortedVideoData[i+1].astype(np.float32)

      spatialRef, temporalRef = extract_info(refFrame1, refFrame2)
      spatialDis, temporalDis = extract_info(disFrame1, disFrame2)

      rreds.append(np.mean(np.abs(spatialRef - spatialDis)))
      rredt.append(np.mean(np.abs(temporalRef - temporalDis)))

      rredssn.append(np.abs(np.mean(spatialRef - spatialDis)))
      rredtsn.append(np.abs(np.mean(temporalRef - temporalDis)))

    rreds = np.array(rreds)
    rredt = np.array(rredt)
    rredssn = np.array(rredssn)
    rredtsn = np.array(rredtsn)

    srred = np.mean(rreds)
    trred = np.mean(rredt)
    srredsn = np.mean(rredssn)
    trredsn = np.mean(rredtsn)

    strred = srred * trred
    strredsn = srredsn * trredsn

    return np.hstack((rreds.reshape(-1, 1), rredt.reshape(-1, 1), rredssn.reshape(-1, 1), rredtsn.reshape(-1, 1))), strred, strredsn


class SpatialSteerablePyramid():
	def __init__(self, height=4):
		"""
        height is the total height, including highpass and lowpass
        """

		self.height = height

	def corr(self, A, fw):
		h, w = A.shape

		sy2 = int(np.floor((fw.shape[0] - 1) / 2))
		sx2 = int(np.floor((fw.shape[1] - 1) / 2))

		# pad the same as the matlabpyrtools
		newpad = np.vstack((A[1:fw.shape[0] - sy2, :][::-1], A, A[h - (fw.shape[0] - sy2):h - 1, :][::-1]))  # ,
		newpad = np.hstack(
			(newpad[:, 1:fw.shape[1] - sx2][:, ::-1], newpad, newpad[:, w - (fw.shape[1] - sx2):w - 1][:, ::-1]))
		newpad = newpad.astype(np.float32)

		return scipy.signal.correlate2d(newpad, fw, mode='valid').astype(np.float32)

	def buildLevs(self, lo0, lofilt, bfilts, edges, mHeight):
		if mHeight <= 0:
			return [lo0]

		bands = []
		for i in range(bfilts.shape[0]):
			filt = bfilts[i]
			bands.append(self.corr(lo0, filt))

		lo = self.corr(lo0, lofilt)[::2, ::2]
		bands = [bands] + self.buildLevs(lo, lofilt, bfilts, edges, mHeight - 1)

		return bands

	def decompose(self, inputimage, filtfile='sp1Filters', edges='symm'):
		inputimage = inputimage.astype(np.float32)

		if filtfile == 'sp5Filters':
			lo0filt, hi0filt, lofilt, bfilts, mtx, harmonics = load_sp5filters()
		else:
			raise (NotImplementedError, "That filter configuration is not implemnted")

		h, w = inputimage.shape

		hi0 = self.corr(inputimage, hi0filt)
		lo0 = self.corr(inputimage, lo0filt)

		pyr = self.buildLevs(lo0, lofilt, bfilts, edges, self.height)
		pyr = [hi0] + pyr

		return pyr

	def extractSingleBand(self, inputimage, filtfile='sp1Filters', edges='symm', band=0, level=1):
		inputimage = inputimage.astype(np.float32)

		if filtfile == 'sp5Filters':
			lo0filt, hi0filt, lofilt, bfilts, mtx, harmonics = load_sp5filters()
		else:
			raise (NotImplementedError, "That filter configuration is not implemnted")

		h, w = inputimage.shape

		if level == 0:
			hi0 = self.corr(inputimage, hi0filt)
			singleband = hi0
		else:
			lo0 = self.corr(inputimage, lo0filt)
			for i in range(1, level):
				lo0 = self.corr(lo0, lofilt)[::2, ::2]

			# now get the band
			filt = bfilts[band]
			singleband = self.corr(lo0, filt)

		return singleband

def vshape(videodata):
    """Standardizes the input data shape.

    Transforms video data into the standardized shape (T, M, N, C), where
    T is number of frames, M is height, N is width, and C is number of
    channels.

    Parameters
    ----------
    videodata : ndarray
        Input data of shape (T, M, N, C), (T, M, N), (M, N, C), or (M, N), where
        T is number of frames, M is height, N is width, and C is number of
        channels.

    Returns
    -------
    videodataout : ndarray
        Standardized version of videodata, shape (T, M, N, C)

    """
    if not isinstance(videodata, np.ndarray):
        videodata = np.array(videodata)

    if len(videodata.shape) == 2:
        a, b = videodata.shape
        return videodata.reshape(1, a, b, 1)
    elif len(videodata.shape) == 3:
        a, b, c = videodata.shape
        # check the last dimension small
        # interpret as color channel
        if c in [1, 2, 3, 4]:
            return videodata.reshape(1, a, b, c)
        else:
            return videodata.reshape(a, b, c, 1)
    elif len(videodata.shape) == 4:
        return videodata
    else:
        raise ValueError("Improper data input")


def load_sp5filters():
  harmonics = np.array([1, 3, 5])

  mtx = np.array([
    [0.3333, 0.2887, 0.1667, 0.0000, -0.1667, -0.2887],
    [0.0000, 0.1667, 0.2887, 0.3333, 0.2887, 0.1667],
    [0.3333, -0.0000, -0.3333, -0.0000, 0.3333, -0.0000],
    [0.0000, 0.3333, 0.0000, -0.3333, 0.0000, 0.3333],
    [0.3333, -0.2887, 0.1667, -0.0000, -0.1667, 0.2887],
    [-0.0000, 0.1667, -0.2887, 0.3333, -0.2887, 0.1667]
  ])

  hi0filt = np.array([
    [-0.00033429, -0.00113093, -0.00171484, -0.00133542, -0.00080639, -0.00133542, -0.00171484, -0.00113093, -0.00033429],
    [-0.00113093, -0.00350017, -0.00243812, 0.00631653, 0.01261227, 0.00631653, -0.00243812, -0.00350017, -0.00113093],
    [-0.00171484, -0.00243812, -0.00290081, -0.00673482, -0.00981051, -0.00673482, -0.00290081, -0.00243812, -0.00171484],
    [-0.00133542, 0.00631653, -0.00673482, -0.07027679, -0.11435863, -0.07027679, -0.00673482, 0.00631653, -0.00133542],
    [-0.00080639, 0.01261227, -0.00981051, -0.11435863, 0.81380200, -0.11435863, -0.00981051, 0.01261227, -0.00080639],
    [-0.00133542, 0.00631653, -0.00673482, -0.07027679, -0.11435863, -0.07027679, -0.00673482, 0.00631653, -0.00133542],
    [-0.00171484, -0.00243812, -0.00290081, -0.00673482, -0.00981051, -0.00673482, -0.00290081, -0.00243812, -0.00171484],
    [-0.00113093, -0.00350017, -0.00243812, 0.00631653, 0.01261227, 0.00631653, -0.00243812, -0.00350017, -0.00113093],
    [-0.00033429, -0.00113093, -0.00171484, -0.00133542, -0.00080639, -0.00133542, -0.00171484, -0.00113093, -0.00033429]
  ])

  lo0filt = np.array([
    [0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614],
    [-0.01551246, 0.05586982, 0.15925570, 0.05586982, -0.01551246],
    [-0.03848215, 0.15925570, 0.40304148, 0.15925570, -0.03848215],
    [-0.01551246, 0.05586982, 0.15925570, 0.05586982, -0.01551246],
    [0.00341614, -0.01551246, -0.03848215, -0.01551246, 0.00341614]
  ])

  lofilt = 2*np.array([
    [0.00085404, -0.00244917, -0.00387812, -0.00944432, -0.00962054, -0.00944432, -0.00387812, -0.00244917, 0.00085404],
    [-0.00244917, -0.00523281, -0.00661117, 0.00410600, 0.01002988, 0.00410600, -0.00661117, -0.00523281, -0.00244917],
    [-0.00387812, -0.00661117, 0.01396746, 0.03277038, 0.03981393, 0.03277038, 0.01396746, -0.00661117, -0.00387812],
    [-0.00944432, 0.00410600, 0.03277038, 0.06426333, 0.08169618, 0.06426333, 0.03277038, 0.00410600, -0.00944432],
    [-0.00962054, 0.01002988, 0.03981393, 0.08169618, 0.10096540, 0.08169618, 0.03981393, 0.01002988, -0.00962054],
    [-0.00944432, 0.00410600, 0.03277038, 0.06426333, 0.08169618, 0.06426333, 0.03277038, 0.00410600, -0.00944432],
    [-0.00387812, -0.00661117, 0.01396746, 0.03277038, 0.03981393, 0.03277038, 0.01396746, -0.00661117, -0.00387812],
    [-0.00244917, -0.00523281, -0.00661117, 0.00410600, 0.01002988, 0.00410600, -0.00661117, -0.00523281, -0.00244917],
    [0.00085404, -0.00244917, -0.00387812, -0.00944432, -0.00962054, -0.00944432, -0.00387812, -0.00244917, 0.00085404]
  ])

  bfilts = np.array([
    [
      [0.00277643, 0.00496194, 0.01026699, 0.01455399, 0.01026699, 0.00496194, 0.00277643],
      [-0.00986904, -0.00893064, 0.01189859, 0.02755155, 0.01189859, -0.00893064, -0.00986904],
      [-0.01021852, -0.03075356, -0.08226445, -0.11732297, -0.08226445, -0.03075356, -0.01021852],
      [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
      [0.01021852, 0.03075356, 0.08226445, 0.11732297, 0.08226445, 0.03075356, 0.01021852],
      [0.00986904, 0.00893064, -0.01189859, -0.02755155, -0.01189859, 0.00893064, 0.00986904],
      [-0.00277643, -0.00496194, -0.01026699, -0.01455399, -0.01026699, -0.00496194, -0.00277643]
    ],
    [
      [-0.00343249, -0.00640815, -0.00073141, 0.01124321, 0.00182078, 0.00285723, 0.01166982],
      [-0.00358461, -0.01977507, -0.04084211, -0.00228219, 0.03930573, 0.01161195, 0.00128000],
      [0.01047717, 0.01486305, -0.04819057, -0.12227230, -0.05394139, 0.00853965, -0.00459034],
      [0.00790407, 0.04435647, 0.09454202, -0.00000000, -0.09454202, -0.04435647, -0.00790407],
      [0.00459034, -0.00853965, 0.05394139, 0.12227230, 0.04819057, -0.01486305, -0.01047717],
      [-0.00128000, -0.01161195, -0.03930573, 0.00228219, 0.04084211, 0.01977507, 0.00358461],
      [-0.01166982, -0.00285723, -0.00182078, -0.01124321, 0.00073141, 0.00640815, 0.00343249]
    ],
    [
      [0.00343249, 0.00358461, -0.01047717, -0.00790407, -0.00459034, 0.00128000, 0.01166982],
      [0.00640815, 0.01977507, -0.01486305, -0.04435647, 0.00853965, 0.01161195, 0.00285723],
      [0.00073141, 0.04084211, 0.04819057, -0.09454202, -0.05394139, 0.03930573, 0.00182078],
      [-0.01124321, 0.00228219, 0.12227230, -0.00000000, -0.12227230, -0.00228219, 0.01124321],
      [-0.00182078, -0.03930573, 0.05394139, 0.09454202, -0.04819057, -0.04084211, -0.00073141],
      [-0.00285723, -0.01161195, -0.00853965, 0.04435647, 0.01486305, -0.01977507, -0.00640815],
      [-0.01166982, -0.00128000, 0.00459034, 0.00790407, 0.01047717, -0.00358461, -0.00343249]
    ],
    [
      [-0.00277643, 0.00986904, 0.01021852, -0.00000000, -0.01021852, -0.00986904, 0.00277643],
      [-0.00496194, 0.00893064, 0.03075356, -0.00000000, -0.03075356, -0.00893064, 0.00496194],
      [-0.01026699, -0.01189859, 0.08226445, -0.00000000, -0.08226445, 0.01189859, 0.01026699],
      [-0.01455399, -0.02755155, 0.11732297, -0.00000000, -0.11732297, 0.02755155, 0.01455399],
      [-0.01026699, -0.01189859, 0.08226445, -0.00000000, -0.08226445, 0.01189859, 0.01026699],
      [-0.00496194, 0.00893064, 0.03075356, -0.00000000, -0.03075356, -0.00893064, 0.00496194],
      [-0.00277643, 0.00986904, 0.01021852, -0.00000000, -0.01021852, -0.00986904, 0.00277643]
    ],
    [
      [-0.01166982, -0.00128000, 0.00459034, 0.00790407, 0.01047717, -0.00358461, -0.00343249],
      [-0.00285723, -0.01161195, -0.00853965, 0.04435647, 0.01486305, -0.01977507, -0.00640815],
      [-0.00182078, -0.03930573, 0.05394139, 0.09454202, -0.04819057, -0.04084211, -0.00073141],
      [-0.01124321, 0.00228219, 0.12227230, -0.00000000, -0.12227230, -0.00228219, 0.01124321],
      [0.00073141, 0.04084211, 0.04819057, -0.09454202, -0.05394139, 0.03930573, 0.00182078],
      [0.00640815, 0.01977507, -0.01486305, -0.04435647, 0.00853965, 0.01161195, 0.00285723],
      [0.00343249, 0.00358461, -0.01047717, -0.00790407, -0.00459034, 0.00128000, 0.01166982]
    ],
    [
      [-0.01166982, -0.00285723, -0.00182078, -0.01124321, 0.00073141, 0.00640815, 0.00343249],
      [-0.00128000, -0.01161195, -0.03930573, 0.00228219, 0.04084211, 0.01977507, 0.00358461],
      [0.00459034, -0.00853965, 0.05394139, 0.12227230, 0.04819057, -0.01486305, -0.01047717],
      [0.00790407, 0.04435647, 0.09454202, -0.00000000, -0.09454202, -0.04435647, -0.00790407],
      [0.01047717, 0.01486305, -0.04819057, -0.12227230, -0.05394139, 0.00853965, -0.00459034],
      [-0.00358461, -0.01977507, -0.04084211, -0.00228219, 0.03930573, 0.01161195, 0.00128000],
      [-0.00343249, -0.00640815, -0.00073141, 0.01124321, 0.00182078, 0.00285723, 0.01166982]
    ]
  ])[:, ::-1, ::-1]

  return lo0filt.astype(np.float32), hi0filt.astype(np.float32), lofilt.astype(np.float32), bfilts.astype(np.float32), mtx.astype(np.float32), harmonics.astype(np.float32)
