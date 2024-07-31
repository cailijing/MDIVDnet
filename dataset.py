"""
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import glob
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from utils import open_sequence

NUMFRXSEQ_VAL = 15	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

class ValDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, valsetdir=None, gray_mode=True, num_input_frames=15, disp_status=True):
		self.gray_mode = gray_mode

		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))

		# open individual sequences and append them to the sequence list
		sequences = []
		seq_dirs_name = []
		for seq_dir in seqs_dirs:
			seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames,disp_status=disp_status)
			# seq is [num_frames, C, H, W]
			if gray_mode == True:
				seq = np.expand_dims(seq, axis=1)

			sequences.append(seq)
			name = seq_dir.split('/')
			seq_dirs_name.append(name[-1])

		self.sequences = sequences
		self.seq_dirs_name = seq_dirs_name


	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index]), self.seq_dirs_name[index]

	def __len__(self):
		return len(self.sequences)


class TestDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, testsetdir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		self.gray_mode = gray_mode

		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(testsetdir, VALSEQPATT)))

		# open individual sequences and append them to the sequence list
		sequences = []
		seq_dirs_name = []
		for seq_dir in seqs_dirs:
			seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
							 max_num_fr=num_input_frames)
			# seq is [num_frames, C, H, W]
			if gray_mode == True:
				seq = np.expand_dims(seq, axis=1)

			sequences.append(seq)
			name = seq_dir.split('/')
			seq_dirs_name.append(name[-1])

		self.sequences = sequences
		self.seq_dirs_name = seq_dirs_name

	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index]), self.seq_dirs_name[index]

	def __len__(self):
		return len(self.sequences)
