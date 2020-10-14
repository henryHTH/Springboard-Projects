
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

def merge_txt_files(all_txt_file,txt_filepath,run_flag=1):


	# create & open a new file in write mode
	
	try:
		with open(txt_filepath,'r', encoding = 'utf_8') as f:
			text = f.read()
			f.close()
			all_txt_file.write(text + '\n')
	except UnicodeDecodeError:
		with open(txt_filepath,'r', encoding = 'windows-1252') as f:
			text = f.read()
			f.close()
			all_txt_file.write(text + '\n')

	# escape newline characters in the original review text
	
#print (u'''Text from {:,} companies risk factors 
#		  written to the new txt file.'''.format(count))




if __name__ == '__main__':

	run_flag = int(input('Run original function:'))
	year = input('Year:')
	data_path = r"/Users/henry/Documents/study/Springboard/nlp_project/data"
	only_dir = [f for f in listdir(data_path) if not isfile(join(data_path, f))]
	all_txt_filepath = join(data_path,f'smaple_text_file_{year}.txt')

	if run_flag:
		
		with open(all_txt_filepath, 'w', encoding='utf_8') as all_txt_file:
			count = 0
			for folder in tqdm(only_dir):
				file_path = join(data_path,folder)
				file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
				for file_name in file_names:
					if year in file_name:
						text = join(file_path,file_name)
						merge_txt_files(all_txt_file,text,run_flag)
						count += 1

				
			all_txt_file.close()
		print (u'Text from {:,} text files in the txt file.'.format(count))

	else:
		count = 0
		with open(all_txt_filepath, encoding='utf_8') as f:
			for count, line in enumerate(f):
				pass
			
		print (u'Text from {:,} restaurant reviews in the txt file.'.format(count + 1))
