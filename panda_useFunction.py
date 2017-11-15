import pandas as pd

class DataMinipulation(object):
	"""docstring for DataMinipulation"""
	def __init__(self):
		super(DataMinipulation, self).__init__()
		self.input_frame = None

	def read_csv(self,filePath):
		self.input_frame = pd.read_csv(filePath)

	#def write_csv(self,data=self.input,filePath,index=False):
	#	data.to_csv(filePath,index=index, encoding='utf-8')

	def covert_cart_asCode(self,col_names=None):
		for i in range(len(col_names)):
			self.input_frame[col_names[i]] = self.input_frame[col_names[i]].astype('category')
			cat_columns = self.input_frame.select_dtypes(['category']).columns
			self.input_frame[cat_columns] = self.input_frame[cat_columns].apply(lambda x: x.cat.codes)
	def get_data(self):
		return self.input_frame
	def del_col(self,col_name):
		self.input_frame = self.input_frame.drop(col_name,1)
	def convert_cart_oneHotEncode(self,col_names=None):
		pass


