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
		
		elif isinstance(key, str):
			name_ind = self.names.index(key)
			return self._rm.getRoi(name_ind)
		
	def open_rois(self, path):
		# load roi from a given path
		full_path = os.path.abspath(path)
		self._rm.open(full_path)
		print("opened rois")
		
	def reset(self):
		self._rm.reset()

	def select(self, pyimp, key):
		# select an roi by name, index or by roi itself
		if isinstance(key, int):
			self._rm.select(pyimp._image, key)

		elif isinstance(key, str):
			ind = self._name_to_ind(key)
			self._rm.select(pyimp._image, ind)

		elif isinstance(key, Roi):
			ind = self._rm.getRoiIndex(key)
			self._rm.select(pyimp._image, ind)

		else:
			raise(ValueError("Unrecognized roi to select."))

	def deselect(self, key=None):

		# deselect all or any single roi
		if isinstance(key, str):
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

	class FromPathHandler(object):

		def __init__(self, PathHandler):
			self.full_path = PathHandler.full_path
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
			self.file = _PathHandler._check_tif(name)
			self.full_path = os.path.join(self.parent, name)

			self.exists = os.path.exists(self.full_path)
			self.is_file = os.path.isfile(self.full_path)
			self.is_dir = os.path.isdir(self.full_path)

			if ret:
				return self


		def __repr__(self):
			return self.full_path




class PyImagePlus(object):
	"""
	A class which is meant to treat imagePlus objects as true objects in python,
	allowing simple manipulation of images without the need to call inconvenient
	ImageJ plugins.
	"""

	_IMAGES = []

	_PyRoiManager = PyRoiManager()	

	@classmethod
	def _GET_N_RECENT_IMAGES(cls, n):
		images = []
		for _ in range(n):
			images.append(WindowManager.getCurrentImage())
			WindowManager.putBehind()
		return images
	
	@classmethod
	def _ADD_IMAGE(cls, obj):
		cls._IMAGES.append(obj)

	@classmethod
	def _REM_IMAGE_BY_TITLE(cls, title):
		for i, cur_img in enumerate(cls._IMAGES):
			if cur_img.title == title:
				ind = i
				break
		del cls._IMAGES[ind]

	
	
	
	
	def __init__(self,
			  image_path = None,
			  _image = None,
			  _path_handler = None):

		# if path given, immortalize it-- else, use None
		if image_path:
			self.image_path = _PathHandler.FromPath(image_path)
			imp = self._load_image()
			self._image = imp

		elif _image:
			imp = _image
			self._image = imp
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
			return self._GET_N_RECENT_IMAGES(1)[0]
	
	@property
	def title(self):
		return self._image.getTitle()	
	@title.setter
	def title(self, value):
		# change the name of the image
		self._image.setTitle(value) 

		# update filename of the image
		if isinstance(self.image_path, _PathHandler.FromPathHandler):
			self.image_path.update_from_name(value, ret=False)

	@property
	def stack_size(self):
		return self._image.getStackSize()

	@property
	def dimensions(self):
		return self._image.getDimensions().tolist()
	
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
	
	def set_title(self, title):
		# for use in method chains
		self.title = title
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
	
	#TODO: fix this: it will return n images as opposed to looping through them individually
	# def __iter__(self):
	#	return iter([self[i+1] for i in range(len(self))])
	
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
		for i in range(self.n_slices):
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
		for img in self._GET_N_RECENT_IMAGES(n):
			out_list.append(PyImagePlus(_image=img, _path_handler=self.image_path))
		out_list.reverse()
		return out_list
	
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
		imgs = self._GET_N_RECENT_IMAGES(n_chans)
		return [PyImagePlus(_image=img, _path_handler=self.image_path) for img in imgs]
		

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
		imgs = self._GET_N_RECENT_IMAGES(n_chans)
		return [
			PyTSeries(
				Fs=self.Fs/n_chans,
				t0_s = self.time_s[i],
				_image=img, 
				_path_handler=self.image_path
				) for i,img in enumerate(imgs)]




#######################
## Import utils #######
#######################
__all__ = ['PyImagePlus', 'PyRoiManager', 'PyTSeries', 'close_all', 'keep_only', 'get_pyimage']