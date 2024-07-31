import os
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types



class VideoReaderPipeline(Pipeline):
	def __init__(self, batch_size, sequence_length, num_threads, device_id, files,
				 crop_size, random_shuffle=True, step=-1, GRAYmode=False):
		super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
		# Define VideoReader
		self.GRAYmode = GRAYmode
		self.reader = ops.VideoReader(device="gpu",
								 filenames=files,
								 sequence_length=sequence_length,
								 normalized=False,
								 random_shuffle=random_shuffle,
								 image_type=types.DALIImageType.RGB,
								 dtype=types.DALIDataType.UINT8,
								 step=step,
								 initial_fill=32)
		self.colorconv = ops.ColorSpaceConversion(device="gpu",
												  image_type=types.RGB,
												  output_type=types.GRAY)
		self.crop = ops.CropMirrorNormalize(device="gpu",
										crop_w=crop_size,
										crop_h=crop_size,
										output_layout='FCHW',
										dtype=types.DALIDataType.FLOAT)
		self.uniform = ops.Uniform(range=(0.0, 1.0))  # used for random crop

	def define_graph(self):
		input = self.reader(name="Reader")
		if self.GRAYmode == True:
			input = self.colorconv(input)
		cropped = self.crop(input, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
		return cropped


class train_dali_loader():
	def __init__(self, batch_size, file_root, sequence_length,
				 crop_size, epoch_size=-1, random_shuffle=True, temp_stride=-1, GRAYmode=True, printfilename=True):
		# Builds list of sequence filenames
		container_files = os.listdir(file_root)
		container_files = [file_root + '/' + f for f in container_files]
		# Define and build pipeline
		if printfilename == True:
			print(container_files)
		# print(container_files)
		self.pipeline = VideoReaderPipeline(batch_size=batch_size,
											sequence_length=sequence_length,
											num_threads=2,
											device_id=0,
											files=container_files,
											crop_size=crop_size,
											random_shuffle=random_shuffle,
											step=temp_stride,
											GRAYmode=GRAYmode)
		self.pipeline.build()

		# Define size of epoch
		if epoch_size <= 0:
			self.epoch_size = self.pipeline.epoch_size("Reader")
		else:
			self.epoch_size = epoch_size
		self.dali_iterator = pytorch.DALIGenericIterator(pipelines=self.pipeline,
														output_map=["data"],
														size=self.epoch_size,
														auto_reset=True)

	def __len__(self):
		return self.epoch_size

	def __iter__(self):
		return self.dali_iterator.__iter__()
