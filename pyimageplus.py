from ij import IJ, WindowManager
from ij.plugin import ImageCalculator, HyperStackConverter, ZProjector, Concatenator
from ij.plugin.frame import RoiManager
from ij.gui import Roi

import os


class PyRoiManager(object):
	"""
	Bindings to ROI manager for easier use.
	"""
	def __init__(self):
		RoiManager()
		self.reset()
	
	@property
	def _rm(self):
		return RoiManager.getRoiManager()

	@property
	def rois(self):
		all_rois = self[0:len(self)]
		return {name : roi for name, roi in zip(self.names, all_rois)}

	@property
	def names(self):
		names = []
		for i in range(len(self)):
			names.append(self._rm.getName(i))
		return names
	
	@property
	def has_selection(self):
		return self._rm.selected() > 0
	
	def _name_to_ind(self, name):
		# utility to return the index of the roi name
		for i, cur_name in enumerate(self.names):
			if cur_name == name:
				return i

	def __len__(self):
		return self._rm.getCount()
	
	def __iter__(self):
		return iter([self[i] for i in range(len(self))])

	def __getitem__(self, key):

		if isinstance(key, int):
			return self._rm.getRoi(key)
		
		elif isinstance(key, slice):
			start, stop, _ = key.start, key.stop, key.step
			out_list = []
			for i in range(start, stop):
				out_list.append(self._rm.getRoi(i))
			return out_list
		
		elif isinstance(key, basestring):
			name_ind = self.names.index(key)
			return self._rm.getRoi(name_ind)
		
	def open_rois(self, path):

		# load roi from a given path
		full_path = os.path.abspath(path)
		self._rm.open(full_path)
		print("Opened rois.")
		
	def reset(self):
		self._rm.reset()

	def select(self, pyimp, key):

		# select an roi by name, index or by roi itself
		if isinstance(key, int):
			self._rm.select(pyimp._image, key)

		elif isinstance(key, basestring):
			ind = self._name_to_ind(key)
			self._rm.select(pyimp._image, ind)

		elif isinstance(key, Roi):
			ind = self._rm.getRoiIndex(key)
			self._rm.select(pyimp._image, ind)

		else:
			raise(ValueError("Unrecognized roi to select."))

	def deselect(self, key=None):

		# deselect all or any single roi
		if isinstance(key, basestring):
			ind = self._name_to_ind(key)
			self._rm.deselect(ind)

		elif isinstance(key, int):
			roi = self._rm.getRoi(key)
			self._rm.deselect(roi)

		elif isinstance(key, Roi):
			self._rm.deselect(key)

		elif isinstance(key, type(None)):
			self._rm.deselect()

		else:
			raise(ValueError("Unrecognized roi to deselect."))
		

class _PathHandler(object):
	"""
	Class to automatically handle pathnames and filenames given the current image title.
	"""

	@staticmethod
	def _check_tif(val):
		if '.tif' not in val:
			val += '.tif'
		return val

	class FromPath(object):

		def __init__(self, path):
			# working with paths
			full_path = os.path.abspath(path)
			self.full_path = full_path
			self._source_path = full_path
			
			self.exists = os.path.exists(full_path)
			self.is_file = os.path.isfile(full_path)
			self.is_dir = os.path.isdir(full_path)
			self.ext = os.path.splitext(full_path)[1]
			
			split = os.path.split(full_path)
			if self.is_file:
				self.parent, self.file = split
			else:
				self.parent, self.file = None, None
			self.is_saved = True

		def __repr__(self):
			return self.full_path
		
		@property
		def source_path(self):
			return self._source_path
		
		@source_path.setter
		def source_path(self, _):
			raise(ValueError("Cannot edit source path."))
		

		def update_from_name(self, name, ret=False):
			"""
			If the name of the PyImagePlus object changes, this is used
			to change the savename within the _PathHandler.FromPathHandler base class.
			"""
			name = _PathHandler._check_tif(name)
			self.file = name
			self.full_path = os.path.join(self.parent, name)
			
			self.exists = os.path.exists(self.full_path)
			self.is_file = os.path.isfile(self.full_path)
			self.is_dir = os.path.isdir(self.full_path)

			if ret:
				return self

	class FromPathHandler(object):

		def __init__(self, PathHandler):
			self.full_path = PathHandler.full_path
			self._source_path = PathHandler.source_path
			self.exists = PathHandler.exists
			self.is_file = PathHandler.is_file
			self.is_dir = PathHandler.is_dir
			self.ext = PathHandler.ext
			self.parent = PathHandler.parent
			self.file = PathHandler.file
			self.is_saved = False

		def update_from_name(self, name, ret=False):
			"""
			If the name of the PyImagePlus object changes, this is used
			to change the savename within the _PathHandler.FromPathHandler base class.
			"""
			name = _PathHandler._check_tif(name)
			self.file = name
			self.full_path = os.path.join(self.parent, name)

			self.exists = os.path.exists(self.full_path)
			self.is_file = os.path.isfile(self.full_path)
			self.is_dir = os.path.isdir(self.full_path)

			if ret:
				return self
			
		@property
		def source_path(self):
			return self._source_path
		
		@source_path.setter
		def source_path(self, _):
			raise(ValueError("Cannot edit source path."))

		def __repr__(self):
			return self.full_path


class _PyWindowManager(object):
	"""
	Helper class to manage windows.
	"""

	@staticmethod
	def _GET_N_RECENT_IMAGES(n):
		images = []
		for _ in range(n):
			images.append(WindowManager.getCurrentImage())
			WindowManager.putBehind()
		return images


class _PyHyperStackIndexer(object):
	"""
	Helper class to handle indexing into a hyperstack.
	order: c, channels
	  		z, slices
			  t, frames

	Must be accessed through properties channels, slices or frames to be indexed.
	"""
	# flat, channel, slice, frame
	# c, z, t

	def __init__(self, pyimp):
		self.pyimp = pyimp
		
	@property
	def channels(self):
		self.dim = "channels"
		return self

	@property
	def slices(self):
		self.dim = "slices"
		return self

	@property
	def frames(self):
		self.dim = "frames"
		return self

	def __getitem__(self, val):

		if isinstance(val, int):
			key = str(val)

		elif isinstance(val, slice):
			start, stop = val.start, val.stop
			key = str(start) + "-" + str(stop)
		
		if self.dim == "channels":
			prompt = "channels=" + key
		elif self.dim == "slices":
			prompt = "slices=" + key
		elif self.dim == "frames":
			prompt = "frames=" + key

		# getting a substack
		IJ.run(self.pyimp._image, "Make Substack...", prompt)

		# collecting recent images
		img = _PyWindowManager._GET_N_RECENT_IMAGES(1)[0]
		return PyImagePlus(_image= img,
					 _path_handler= self.pyimp.image_path).set_title(self.pyimp.title + " " + prompt)


class PyImagePlus(object):
	"""
	A class which is meant to treat imagePlus objects as true objects in python,
	allowing simple manipulation of images without the need to call inconvenient
	ImageJ plugins.
	"""

	_IMAGES = []

	_PyRoiManager = PyRoiManager()	
	
	@classmethod
	def _ADD_IMAGE(cls, obj):
		cls._IMAGES.append(obj)

	@classmethod
	def _REM_IMAGE_BY_TITLE(cls, title):
		for i, cur_img in enumerate(cls._IMAGES):
			if cur_img.title == title:
				del cls._IMAGES[i]
				return

	def __init__(self,
			  image_path = None,
			  _image = None,
			  _path_handler = None):

		# if path given, immortalize it-- else, use None
		if image_path:
			self.image_path = _PathHandler.FromPath(image_path)
			self._image = self._load_image()

		elif _image:
			# _path_handler being passed in implied
			self._image = _image
			self.image_path = _PathHandler.FromPathHandler(_path_handler).update_from_name(self.title, True)
			
		# showing the image after initialization
		self._image.show()
				
		# automatically enhancing contrast based on entire image
		self.select_all()
		self.enhance_contrast()

		# handle protected propreties & add image to the master list
		self._window = self._image.getWindow()
		self._processor = self._image.getProcessor()
		self._ADD_IMAGE(self)

		# roi manager
		self.roi_path = None
		self.cur_roi = "All"

	def _load_image(self):
		if self.image_path.ext =='.tif':
			print("Opening .tif file")
			return IJ.openImage(self.image_path.full_path)
		elif self.image_path.ext =='.pcoraw':
			print("Opening .pcoraw file")
			IJ.run("Bio-Formats Importer", "open=" + self.image_path.full_path + " autoscale color_mode=Default rois_import=[ROI manager] view=[Standard ImageJ] stack_order=Default")
			return _PyWindowManager._GET_N_RECENT_IMAGES(1)[0]
	
	@property
	def title(self):
		return self._image.getTitle()	
	@title.setter
	def title(self, value):
		# change the name of the image
		self._image.setTitle(value) 

		# update filename of the image
		self.image_path.update_from_name(value, ret=False)

	@property
	def stack_size(self):
		return self._image.getStackSize()

	@property
	def dimensions(self):
		return self._image.getDimensions().tolist()
	@dimensions.setter
	def dimensions(self, val):
		if len(val) != 3:
			raise(ValueError("Setting dimensions requires [channels slices frames], use a list."))
		self._image.setDimensions(val[0], val[1], val[2])

	@property
	def index(self):
		return self._image.getSlice()
	@index.setter
	def index(self, val):
		self._image.setSlice(val)
	
	@property
	def n_channels(self):
		return self._image.getNChannels()
	
	@property
	def n_dimensions(self):
		return self._image.getNDimensions()
	
	@property
	def n_frames(self):
		return self._image.getNFrames()
	
	@property
	def n_slices(self):
		return self._image.getNSlices()
	
	@property
	def rois(self):
		return self._PyRoiManager.rois
	
	@property
	def is_hyperstack(self):
		return self._image.isDisplayedHyperStack()
	
	@property
	def is_stack(self):
		return not self.is_hyperstack
	
	@property
	def channels(self):
		#returns an object with indexing capabilites
		return _PyHyperStackIndexer(self).channels
	
	@property
	def slices(self):
		return _PyHyperStackIndexer(self).slices
	
	@property
	def frames(self):
		return _PyHyperStackIndexer(self).frames
	
	def set_title(self, title):
		# for use in method chains
		self.title = title
		return self
	
	def set_dimensions(self, n_chans, n_slices, n_frames):
		self.dimensions = [n_chans, n_slices, n_frames]
		return self

	def _dup_image(self):
		new_imp = self._image.duplicate()
		return new_imp
	
	def duplicate(self):
		new_imp = self._image.duplicate()
		return PyImagePlus(_image=new_imp, _path_handler=self.image_path)

	def close(self):
		self._image.close()

	def hide(self):
		self._image.hide()

	def show(self):
		self._image.show()

	####################################
	## Hyperstack & stack conversions ##
	####################################

	def to_hyperstack(self):
		if self.is_hyperstack:
			return self
		self.deselect_roi()
		self._image = HyperStackConverter.toHyperStack(self._image, 
												 self.n_channels,
												 self.n_slices,
												 self.n_frames,
												 "grayscale")
		return self

	def to_stack(self):
		if self.is_stack:
			return self	
		self.deselect_roi()
		self._image = HyperStackConverter.toStack(self._image)
		return self

	#####################
	# binary operations #
	#####################

	def __op2(self, other, operation):
		# genenic function to perform simple operations on other images or on numbers
			
		if isinstance(other, PyImagePlus):

			operation += " create 32-bit stack"
			imp = ImageCalculator.run(self._image, other._image, operation)
			return PyImagePlus(_image=imp, _path_handler=self.image_path)
				
		elif isinstance(other, int) or isinstance(other, float):

			# duplicate image, perform operation on that image and then return
			imp = self._dup_image()
			IJ.run(imp, operation + "...", "value=" + str(other) + " stack")
			return PyImagePlus(_image=imp, _path_handler=self.image_path)

		else:
			raise(ValueError(operation + " not defined for object of type " + str(type(other))))		
		
	def __op1(self, operation, inplace=False):
		if inplace:
			IJ.run(self._image, operation, "stack")
		else:
			imp = self._dup_image()
			IJ.run(imp, operation, "stack")
			return PyImagePlus(_image=imp, _path_handler=self.image_path)
		
	def __repr__(self):
		return self.title + " (roi: " + str(self.cur_roi) + ")"
	
	def __abs__(self):
		return self.__op1("Abs")
	
	def __pow__(self, other):
		# custom code to make power
		IJ.run(self._image, "32-bit", "")

		imp = self._dup_image()
		IJ.run(imp, "Macro...", "code=v^" + str(other) + " stack")
		return PyImagePlus(_image=imp, _path_handler=self.image_path)
	
	def __neg__(self):
		return self.__op2(-1, "Multiply")

	def __add__(self, other):
		return self.__op2(other, "Add")	
	
	def __sub__(self, other):
		return self.__op2(other, "Subtract")
	
	def __div__(self, other):
		return self.__op2(other, "Divide")
	
	def __mul__(self, other):
		return self.__op2(other, "Multiply")
	
	def __and__(self, other):
		return self.__op2(other, "AND")
	
	def __or__(self, other):
		return self.__op2(other, "OR")
	
	def __xor__(self, other):
		return self.__op2(other, "XOR")
	
	def __radd__(self, other):
		return self + other
	
	def __rsub__(self, other):
		return -self + other
	
	def __rmul__(self, other):
		return self * other
	
	def __rdiv__(self, other):
		self._processor.invert()
		out = self * other
		self._processor.invert()
		return out

	##########################
	# Indexing and iteration #
	##########################
	def __getitem__(self, key):
		
		if isinstance(key, slice):
			start, stop, _ = key.start, key.stop, key.step
			imp = self._image.crop(str(start) + "-" + str(stop))
			return PyImagePlus(_image = imp, _path_handler=self.image_path)
		
		elif isinstance(key, int):

			if key == 0:
				raise(ValueError("Indexing at 0 is not supported; start indexing at 1."))
			
			imp = self._image.crop(str(key))
			return PyImagePlus(_image = imp, _path_handler=self.image_path)
		
	def __len__(self):
		return self.stack_size
	
	#FIXME: fix this: it will return n images as opposed to looping through them individually
	# def __iter__(self):
	#	return iter([self[i+1] for i in range(len(self))])

	###############
	# File saving #
	###############
	def save(self, savepath=None):
		if savepath:
			IJ.saveAs(self._image, "Tiff", os.path.abspath(savepath))
		else:
			IJ.saveAs(self._image, "Tiff", self.image_path)
		print("Wrote" + self.image_path)
	
	#####################
	# Custom operations #
	#####################

	# custom operations
	def log(self):
		IJ.run(self._image, "Log", "stack")
		return self

	def exp(self):
		IJ.run(self._image, "Exp", "stack")
		return self
		
	def sqrt(self):
		IJ.run(self._image, "Square Root", "stack")
		return self
	
	def invert(self):
		IJ.run(self._image, "Invert", "stack")
		return self
	
	def square(self):
		IJ.run(self._image, "Square", "stack")
		return self
	
	##############
	# Statistics #
	##############
	
	def _get_stats(self, all=False):
		"""
		Helper function to only use getAllStatistics when asked, as it
		is much slower. Otherwise, use getStatistics.
		"""
		cur_slice = self._image.getSlice()
		out = []
		for i in range(self.stack_size):
			self._image.setSlice(i+1)
			if all:
				out.append(self._image.getAllStatistics())
			else:
				out.append(self._image.getStatistics())
		self._image.setSlice(cur_slice)
		return out
	
	@property
	def _all_stats(self):
		return self._get_stats(True)

	@property
	def stats(self):
		return self._get_stats()

	@property
	def mean(self):
		return [x.mean for x in self.stats]			
	
	@property
	def std(self):
		return [x.stdDev for x in self._all_stats]			
			
	@property
	def median(self):
		return [x.median for x in self._all_stats]		

	@property
	def mode(self):
		return [x.mode for x in self._all_stats]	
	
	@property
	def kurtosis(self):
		return [x.kurtosis for x in self._all_stats]
	
	@property
	def skewness(self):
		return [x.skewness for x in self._all_stats]
	
	@property
	def area(self):
		return [x.area for x in self._all_stats]
	
	@property
	def max(self):
		return [x.max for x in self._all_stats]
	
	@property
	def min(self):
		return [x.min for x in self._all_stats]	


	##########################
	# Container manipulation #
	##########################
		
	def concatenate(self, other):
		"""
		concatenate stacks, but not in place
		"""
		imp = Concatenator.run(self._dup_image(), other._dup_image())
		imp.show()
		if imp.isHyperStack():
			HyperStackConverter.toStack(imp)
		imp.title = self.title + ", " + other.title
		return PyImagePlus(_image=imp, _path_handler=self.image_path)
	
	def split(self, n):
		"""
		splits current stack into other stacks
		"""
		IJ.run(self._image, "Stack Splitter", "number="+str(n))
		out_list = []
		for img in _PyWindowManager._GET_N_RECENT_IMAGES(n):
			out_list.append(PyImagePlus(_image=img, _path_handler=self.image_path))
		out_list.reverse()
		return out_list
	
	###############################
	# Miscellaneous functionality #
	###############################

	def plot(self):
		"""
		Create a plot image.
		"""
		
		IJ.run(self._image, "Plot Z-axis Profile", "")
		result = WindowManager.getCurrentImage()
		result.show()

		return PyImagePlus(_image=result, _path_handler=self.image_path)
	
	def z_project(self, stat):
		"""
		complete a z-projection using the statistic given
		"""
		stats = ['avg', 'min', 'max', 'sum', 'sd', 'median']
		if stat not in stat:
			raise(ValueError("statistic " + str(stat) + " is not valid. Please use one of \n\t" + str(stats)))
		result = ZProjector.run(self._image, stat)
		return PyImagePlus(_image=result, _path_handler=self.image_path)
	
	def deinterleave(self, n_chans):
		IJ.run(self._image, "Deinterleave", "how=" + str(n_chans))
		imgs = _PyWindowManager._GET_N_RECENT_IMAGES(n_chans)
		return [PyImagePlus(_image=img, _path_handler=self.image_path) for img in imgs]

	def combine(self, other, vert=False):
		"""
		runs ImageJ's stack combine function.
		Warning: this is destructive, and images being combined will individually
		lose their references.
		"""		
		command = "stack1=" + self.title + " stack2=" + other.title
		if vert:
			command += " combine"
		IJ.run("Combine...", command)
		img = _PyWindowManager._GET_N_RECENT_IMAGES(1)[0]

		#self._REM_IMAGE_BY_TITLE(self.title)
		#self._REM_IMAGE_BY_TITLE(other.title)

		return PyImagePlus(_image=img, _path_handler=self.image_path).set_title(self.title + other.title)
	
	def z_score(self):
		"""
		Returns the z-score of the image stack.
		"""
		imp_z = (self - self.z_project("avg")) / self.z_project("sd")
		return imp_z
	
	def percentile(self):
		"""
		Returns the image in terms of percentile.
		"""
		imp_perc = (self - self.z_project('min')) / (self.z_project('max') - self.z_project('min'))
		return imp_perc
	
	def df_f0(self, f0):
		"""
		Returns df/f0 normalization of the given dataset
		"""
		if isinstance(f0, int):
			# normalize to said frame
			pyimp_dff0 = (self - self[f0]) / self[f0]

		elif isinstance(f0, PyImagePlus):
			pyimp_dff0 = (self - f0) / f0

		elif isinstance(f0, basestring):
			# try to get the statistic
			val = self.z_project(f0)
			pyimp_dff0 = (self - val) / val

		else:
			raise(ValueError("Unknown input for df_f0: " + str(f0)))
		
		return pyimp_dff0
	
	def cv(self):
		"""
		Returns the coefficient of variation of the image.
		"""
		return self.z_project("sd") / self.z_project('avg')
	
	def snr(self):
		"""
		Returns a rough estimate of signal to noise ratio, defined as mean/sd.
		"""
		return self.z_project("avg") / self.z_project('sd')


	##################
	## ROI utilites ##
	##################
	def select_roi(self, val):			
		self._PyRoiManager.select(self, val)
		self.cur_roi = val
		return self

	def deselect_roi(self, val=None):
		self._PyRoiManager.deselect(val)
		self.cur_roi = None
		return self
	
	def select_all(self):
		IJ.run(self._image, "Select All", "")
		self.cur_roi = "All"
		return self

	#################
	## Filters ######
	#################
	def filt_gauss3D(self, x=0, y=0, z=3):
		IJ.run(self._image, "Gaussian Blur 3D...", "x=" + str(x) + " y=" + str(y) + " z=" + str(z))
		return self
	
	# TODO: add more filters here

	#################
	# B & C #########
	#################
	def enhance_contrast(self):
		IJ.run(self._image, "Enhance Contrast", "saturated=0.35")
		return self


####################
# Window utilities #
####################

def close_all():
	IJ.run("Close All")

def keep_only(*imgs):
	"""
	Closes all images except for those in the function.
	"""
	all_titles = WindowManager.getImageTitles()
	keep_titles = [img.title for img in imgs]
	for title in all_titles:
		if title in keep_titles:
			continue
		imp = WindowManager.getImage(title)
		imp.close()
		PyImagePlus._REM_IMAGE_BY_TITLE(title)

def get_pyimage(title):
	for img in PyImagePlus._IMAGES:
		if img.title == title:
			WindowManager.toFront(img._window)
			return img



class PyTSeries(PyImagePlus):

	"""
	A wrapper around PyImagePlus to handle time series images.
	Enables easier slicing of images wrt timed events.
	"""

	def __init__(self,
			  image_path = None,
			  Fs = 60,
			  t0_s = 0,
			  _image = None,
			  _path_handler = None):
			
			super(PyTSeries, self).__init__(image_path, _image, _path_handler)

			if not self._image:
				raise(ValueError("Image unavailable."))
			
			# adding time-related attributes
			self.Fs = float(Fs)
			self.t0_s = t0_s
			per = 1000/Fs # period in ms
			self.time_ms = [float((i*per) + (t0_s*1000)) for i in range(len(self))]
			self.time_s = [x/1000 for x in self.time_ms]
			self.time_range = (min(self.time_ms), max(self.time_ms))

	def __getitem__(self, key):
		
		if isinstance(key, slice):
			start, stop, _ = key.start, key.stop, key.step
			imp = self._image.crop(str(start) + "-" + str(stop))
			return PyTSeries(
				Fs=self.Fs,
				t0_s = self.time_s[start],
				_image = imp, 
				_path_handler=self.image_path
				)
		
		elif isinstance(key, int):

			if key == 0:
				raise(ValueError("Indexing at 0 is not supported; start indexing at 1."))
			
			imp = self._image.crop(str(key))
			return PyTSeries(
				Fs=self.Fs,
				t0_s = self.time_s[start],
				_image = imp, 
				_path_handler=self.image_path
				)	

	def time_slice(self, start_ms, stop_ms):
		"""
		Slices the image between two time points.
		"""
		start_ind = [x > start_ms for x in self.time_ms].index(1)
		stop_ind = [x <= stop_ms for x in self.time_ms].index(0)
		return self[start_ind:stop_ind]

	def deinterleave(self, n_chans):
		IJ.run(self._image, "Deinterleave", "how=" + str(n_chans))
		imgs = _PyWindowManager._GET_N_RECENT_IMAGES(n_chans)
		return [
			PyTSeries(
				Fs=self.Fs/n_chans,
				t0_s = self.time_s[i],
				_image=img, 
				_path_handler=self.image_path
				) for i,img in enumerate(imgs)]




class PyImageMatrix(object):
	"""
	A class to define array-like operations over python images.
	These are explicitly 2D arrays. Vectors, nor nd-arrays are allowed.
	"""
	_ACCEPTED_CLASSES = [PyImagePlus, PyTSeries]
	_ACCEPTED_CLASS_NAMES = [x.__name__ for x in _ACCEPTED_CLASSES]

	@staticmethod
	def _unique(x):
		first = x[0]
		return all([x==first for x in x[1:]])

	@classmethod
	def _assert_class_acceptable(cls, obj):
		in_valid_options = [isinstance(obj, x) for x in cls._ACCEPTED_CLASSES]
		if not any(in_valid_options):
			raise(ValueError("Cannot make PyImageArray from object " + str(obj.__class__.__name__)))
	
	@classmethod
	def _assert_same_class(cls, x):
		classes = [z.__class__.__name__ for y in x for z in y]
		if not cls._unique(classes):
			raise(ValueError("Objects in PyImageArray array are not of the same class."))
		
	@classmethod
	def _assert_2D(cls, x):
		# check for 2D
		try:
			x[0]
			x[0][0]
		except:
			raise(ValueError("Given array is not 2D."))

		# check not 3D
		# should be able to index into the image, but shouldn't be a list.
		if x[0][0][1].__class__.__name__ not in cls._ACCEPTED_CLASS_NAMES:
			raise(ValueError("3D array detected- please use a 2D array."))
		
	@staticmethod
	def _assert_not_ragged(x):
		lengths = [len(y) for y in x]
		if len(set(lengths)) != 1:
			raise(ValueError("Ragged array passed."))
		
	@classmethod
	def _assert_uniform_dims(cls, x):
		dims = [z.dimensions for y in x for z in y]
		if not cls._unique(dims):
			raise(ValueError("Image objects are of unequal dimensions."))
		
	@staticmethod
	def _flatten_2d(x):
		return [z for y in x for z in y]

	def __init__(self, 
			  pyimps,
			  _make_image = True):
		
		# assertion list for proper input
		for pyimp in self._flatten_2d(pyimps):
			self._assert_class_acceptable(pyimp)
		self._assert_same_class(pyimps)
		self._assert_2D(pyimps)
		self._assert_not_ragged(pyimps)
		self._assert_uniform_dims(pyimps)
			

		# set array and file object values
		self._shape = [len(pyimps), len(pyimps[0])] + pyimps[0][0].dimensions
		self._flat = self._flatten_2d(pyimps) # no need to save ragged, just save flat along with dimensions
		self._image_path_list = [x.image_path for x in self._flat]
		self._titles = [x.title for x in self._flat]
		self._is_T = False

		if _make_image:
			# if image is made, operations on elements are off-limits
			self._image = self._make_matrix_pyimp(self.n_rows, self.n_cols)
			self._image.show()
		else:
			self._image = None	

	@property
	def shape(self):
		return tuple(self._shape)

	@property
	def n_rows(self):
		return self._shape[0]
	
	@property
	def n_cols(self):
		return self._shape[1]
	
	@property
	def n_elements(self):
		return self.n_rows * self.n_cols
		
	def _make_matrix_pyimp(self, n_rows, n_cols, T=False):
		"""
		used to make the PyImagePlus object expressing these images
		this is destructive, i.e. the original images no longer exist after
		calling combine.

		As such, the image made here relies on duplicates of the images.
		"""

		# duplicates
		if T:
			dups = [x.duplicate() for x in self._flat_T]
			self._is_T = not self._is_T
		else:
			dups = [x.duplicate() for x in self._flat]

		# have to form each row independently
		rows = []
		for i in range(n_rows):
			for j in range(n_cols):
				ind = self._get_flat_ind(self.shape, i,j)
				if j == 0:
					cur_img = dups[ind]
					continue
				cur_img = cur_img.combine(dups[ind])
			rows.append(cur_img)

		for i,row in enumerate(rows):
			if i == 0:
				cur_row = row
				continue
			cur_row = cur_row.combine(row, vert=True)
		
		return cur_row
	
	@staticmethod
	def _get_flat_ind(shape, row, col):
		# shape: the shape of the non-flat list
		# row: the row index of the non-flat list
		# col: the col index of the non-flat list
		# returns the index of the flattened list
		return (shape[1]*row) + col

	@property
	def _flat_T(self):
		"""
		Return the flat list in an order for easy
		indexing to create the image
		"""
		inds = []
		for r in range(self.n_rows):
			inds += [r+(i*self.n_rows) for i in range(self.n_cols)]
		return [self._flat[i] for i in inds]
	
	@property
	def T(self):
		# copying old shape
		cur_shape = self._shape[:]

		# adjusting new shape
		self._shape[0] = cur_shape[1]
		self._shape[1] = cur_shape[0]

		# closing the old image and opening a new one
		self._image.close()
		self._image = self._make_matrix_pyimp(self.n_rows, self.n_cols, T=True)
		self._image.show()

	def as_pyimp(self):
		# convert back to PyImagePlus to have access to all stats tools for this object
		return PyImagePlus(
			_image = self._image._image,
				_path_handler = self._image_path_list[0])
	
	#TODO: get for individual images to crop them out and save them

	

#######################
## Import utils #######
#######################
__all__ = ['PyImagePlus', 'PyRoiManager', 'PyTSeries', 'PyImageMatrix', 'close_all', 'keep_only', 'get_pyimage']