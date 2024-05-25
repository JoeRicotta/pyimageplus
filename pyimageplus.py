from ij import IJ, WindowManager
from ij.plugin import ImageCalculator, HyperStackConverter, ZProjector, Concatenator
from ij.plugin.frame import RoiManager
from ij.gui import Roi
from ij.measure import ResultsTable

import os
import json

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


class PyImagePlus(object):

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

	_METADATA_DEFAULT = {"json" : dict(),
					  "mat" : dict(),
					  "tdms" : dict(),
					  "tif" : dict(),
					  "bin" : dict(),
					  "pcoraw" : dict(),
					  "csv" : dict(),
					  "txt" : dict()}
	
	_SAVED_ATTRIBUTES = ["name",
					  "path",
					  "image_path",
					  "metadata_paths"] # for use in saving json object

	"""
	A class which is meant to treat imagePlus objects as true objects in python,
	allowing simple manipulation of images without the need to call inconvenient
	ImageJ plugins.
	"""
    
	def __init__(self,
			  image_path = None,
			  metadata_paths = _METADATA_DEFAULT,
			  _image = None):

		# if path given, immortalize it-- else, use None
		if image_path:
			full_path = os.path.abspath(image_path)
			imp = IJ.openImage(full_path)
			self._image = imp
			self._image_path = full_path

		elif _image:
			imp = _image
			self._image_path = image_path
			self._image = imp

		# private variables
		self._metadata_paths = metadata_paths
		self._path = None # if saved object, where is json file

		# showing the image after initialization
		self._image.show()
				
		# automatically enhancing contrast
		IJ.run(self._image, "Select All", "")
		IJ.run(self._image, "Enhance Contrast", "saturated=0.35")

		# handle protected propreties & add image to the master list
		self._window = self._image.getWindow()
		self._processor = self._image.getProcessor()
		self._ADD_IMAGE(self)

		# roi manager
		self.roi_path = None

	@property
	def image_path(self):
		return self._image_path
	@property
	def metadata_paths(self):
		return self._metadata_paths
	@property
	def path(self):
		return self._path
	
	@property
	def title(self):
		return self._image.getTitle()	
	@title.setter
	def title(self, value):
		self._image.setTitle(value)

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
	
	def add_metadata(self, key, file):
		"""
		Add metadata to the object using the given key and file extension
		"""
		# identify the extension of the file
		full_file = os.path.abspath(file)
		_, file_ext = os.path.splitext(full_file)

		# ensure the file extension is in the provided dictionary
		if not self._metadata_paths.get(file_ext):
			raise(ValueError("File extension " + file_ext + " not recognized."))
		
		# check the metadata dictionary for the key
		if self._metadata_paths[file_ext].get(key):
			self._metadata_paths[file_ext][key].append(full_file)
		else:
			self._metadata_paths[file_ext].update({key : [full_file]})

	def _dup_image(self):
		new_imp = self._image.duplicate()
		return new_imp
	
	def duplicate(self):
		new_imp = self._image.duplicate()
		return PyImagePlus(_image=new_imp)

	def close(self):
		self._image.close()

	def hide(self):
		self._image.hide()

	def show(self):
		self._image.show()

	def save(self, save_path=None):
		return 
		"""
		Save a file with all relevant string information from this object,
		as well as the ImagePlus object it represents
		"""
		if save_path:
			# coercing to full path
			full_path = os.path.abspath(save_path)
		else:
			# extract the path from the image_path and current name
			full_path = self.image_path

		# checking to ensure .json is the extension
		_, ext = os.path.splitext(full_path)
		if ext != ".json":
			raise(ValueError("Saved object must have .json extension, instead got " + ext))
		
		# gather all relevant attributes from this object
		out_dict = dict()
		for attribute in self._SAVED_ATTRIBUTES:
			out_dict.update({attribute : self.__dict__.get(attribute)})

		# prior to saving, alerting to any previous save
		if os.path.exists(full_path):
			print("Overwriting previous file.")

		# saving the dictionary as json
		stream = open(full_path, "w")
		json.dump(out_dict, stream, indent='\t', sort_keys=True)
		
		# adding save path to object
		self._path = full_path

		# now saving ImageJ
		IJ.save()

	#####################
	# binary operations #
	#####################

	def __op2(self, other, operation):
		# genenic function to perform simple operations on other images or on numbers
			
		if isinstance(other, PyImagePlus):

			operation += " create 32-bit stack"
			imp = ImageCalculator.run(self._image, other._image, operation)
			return PyImagePlus(_image=imp)
				
		elif isinstance(other, int) or isinstance(other, float):

			# duplicate image, perform operation on that image and then return
			imp = self._dup_image()
			IJ.run(imp, operation + "...", "value=" + str(other) + " stack")
			return PyImagePlus(_image=imp)

		else:
			raise(ValueError(operation + " not defined for object of type " + str(type(other))))		
		
	def __op1(self, operation, inplace=False):
		if inplace:
			IJ.run(self._image, operation, "stack")
		else:
			imp = self._dup_image()
			IJ.run(imp, operation, "stack")
			return PyImagePlus(_image=imp)
		
	def __repr__(self):
		return self.title
	
	def __abs__(self):
		return self.__op1("Abs")
	
	def __pow__(self, other):
		# custom code to make power
		IJ.run(self._image, "32-bit", "")

		imp = self._dup_image()
		IJ.run(imp, "Macro...", "code=v^" + str(other) + " stack")
		return PyImagePlus(_image=imp)
	
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
	
	def __max__(self):
		return self._processor.getMax()
	
	def __min__(self):
		return self._processor.getMin()		

	##########################
	# Indexing and iteration #
	##########################
	def __getitem__(self, key):

		if isinstance(key, slice):
			start, stop, _ = key.start, key.stop, key.step
			imp = self._image.crop(str(start) + "-" + str(stop))
			return PyImagePlus(_image = imp)
		
		elif isinstance(key, int):
			imp = self._image.crop(str(key))
			return PyImagePlus(_image = imp)
		
	def __len__(self):
		return self.dimensions[3]
	
	def __iter__(self):
		return iter([self[i] for i in range(len(self))])
	
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
		return PyImagePlus(_image=imp)
	
	def split(self, n):
		"""
		splits current stack into other stacks
		"""
		IJ.run(self._image, "Stack Splitter", "number="+str(n))
		out_list = []
		for img in self._GET_N_RECENT_IMAGES(n):
			out_list.append(PyImagePlus(_image=img))
		out_list.reverse()
		return out_list
	
	def plot(self):
		"""
		Create a plot image.
		"""
		
		IJ.run(self._image, "Plot Z-axis Profile", "")
		result = WindowManager.getCurrentImage()
		result.show()

		return PyImagePlus(_image=result)
	
	def z_project(self, stat):
		"""
		complete a z-projection using the statistic given
		"""
		stats = ['avg', 'min', 'max', 'sum', 'sd', 'median']
		if stat not in stat:
			raise(ValueError("statistic " + str(stat) + " is not valid. Please use one of \n\t" + str(stats)))
		result = ZProjector.run(self._image, stat)
		return PyImagePlus(_image=result)

	def measure(self):

		if not self._PyRoiManager.has_selection:
			IJ.run(self._image, "Select All", "")
		
		rt = ResultsTable()
		res = self._PyRoiManager._rm.multiMeasure(self._image)
		rt.updateResults()
		return res
	
	def deinterleave(self, n_chans):
		IJ.run(self._image, "Deinterleave", "how=" + str(n_chans))
		imgs = self._GET_N_RECENT_IMAGES(n_chans)
		return [PyImagePlus(_image=img) for img in imgs]
		

	##################
	## ROI utilites ##
	##################
	def select_roi(self, val):
		self._PyRoiManager.select(self, val)
		return self

	def deselect_roi(self, val=None):
		self._PyRoiManager.deselect(val)
		return self

	#################
	## Filters ######
	#################
	def filt_gauss3D(self, x=0, y=0, z=3):
		IJ.run(self._image, "Gaussian Blur 3D...", "x=" + str(x) + " y=" + str(y) + " z=" + str(z))
		return self
	
	# TODO: add more filters here


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

#######################
## Import utils #######
#######################
__all__ = ['PyImagePlus', 'PyRoiManager', 'close_all', 'keep_only', 'get_pyimage']