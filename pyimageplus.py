from ij import IJ, WindowManager, ImagePlus
from ij.plugin import ImageCalculator, HyperStackConverter, ZProjector, Concatenator
from ij.plugin.frame import RoiManager

import os
import json

class PyImagePlus(object):

	@classmethod
	def _GET_N_RECENT_IMAGES(cls, n):
		images = []
		for _ in range(n):
			images.append(WindowManager.getCurrentImage())
			WindowManager.putBehind()
		return images

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

	@property
	def image_path(self):
		return self._image_path
	@property
	def metadata_paths(self):
		return self._metadata_paths
	@property
	def path(self):
		return self._path
	
	# name attribute: extract from image
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

	def _select_image(self):
		WindowManager.getImage(self.title)

	#####################
	# binary operations #
	#####################

	def __op2(self, other, operation, inplace=False, enforce32=True):
		# genenic function to perform simple operations on other images or on numbers
		if enforce32:
			IJ.run(self._image, "32-bit", "")
			
		if isinstance(other, PyImagePlus):
			if enforce32:
				IJ.run(other._image, "32-bit", "")

			if inplace:
				ImageCalculator.run(self._image, other._image, operation)				

			else:	
				operation += " create 32-bit stack"
				imp = ImageCalculator.run(self._image, other._image, operation)
				return PyImagePlus(_image=imp)
				
		elif isinstance(other, int) or isinstance(other, float):
			if inplace:
				# edit image in-place using IJ.run
				IJ.run(self._image, operation + "...", "value=" + str(other) + " stack")
			else:
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
	
	def __iadd__(self, other):
		return self.__op2(other, "Add", inplace=True)

	def __isub__(self, other):
		return self.__op2(other, "Subtract", inplace=True)
	
	def __imul__(self, other):
		return self.__op2(other, "Multiply", inplace=True)
	
	def __idiv__(self, other):
		return self.__op2(other, "Divide", inplace=True)
	
	def __radd__(self, other):
		return self + other
	
	def __rsub__(self, other):
		return -self + other
	
	def __rmul__(self, other):
		return self * other
	
	def __rdiv__(self, other):
		self.invert()
		out = other * self
		self.invert()
		return out

	def invert(self):
		IJ.run(self._image, "32-bit", "")
		inverted = self._image.getProcessor()
		inverted.invert()
		self._image.setProcessor(inverted)

	##########################
	# Indexing and iteration #
	##########################
	def __getitem__(self, key):

		if isinstance(key, slice):
			start, stop, by = key.start, key.stop, key.step
			imp = self._image.crop(str(start) + "-" + str(stop))
			return PyImagePlus(_image = imp)
		
		elif isinstance(key, int):
			imp = self._image.crop(str(key))
			return PyImagePlus(_image = imp)
		
	def __len__(self):
		return self.dimensions[3]
	
	#####################
	# Custom operations #
	#####################

	# custom operations
	def _log(self):
		pass

	def _exp(self):
		pass

	def append(self, other):
		"""
		build a new stack in-place
		"""
		imp = Concatenator.run(self._image, other._image)
		imp.show()
		if imp.isHyperStack():
			HyperStackConverter.toStack(imp)
		
		self._image.close()
		self._image = imp


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

	def plot(self):
		"""
		Create a plot image.
		"""
		IJ.run(self._image, "Plot Z-axis Profile", "")
		result = WindowManager.getCurrentImage()
		result.show()

		return PyImagePlus(_image=result)
	
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





def close_all():
	IJ.run("Close All")
		
def log(pmp):
	"""
	log for a pyImagePlus object.
	"""
	return pmp._log()
		

		


		

	

		

