import pandas as pd
import os
import chardet
from multiprocessing import Pool


def process_file(file_path):
    # Detects the encoding of the file
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

        # Read txt file
    data = pd.read_csv(file_path, sep='\s+', header=0, encoding=encoding)  # 使用检测到的编码

    # Gets the file name (minus.txt) as the Excel file name and new table header
    filename = os.path.basename(file_path).replace('.txt', '')
    data['Well name'] = filename  # Adds the file name as a new column

    # Reorder the data to make sure "Well name" is listed in the first column
    data = data[['Well name'] + [col for col in data.columns if col != 'Well name']]

    # Get the directory path of the txt file and use it as the save path for the Excel file
    directory = os.path.dirname(file_path)
    excel_filename = os.path.join(directory, filename + '.xlsx')
    data.to_excel(excel_filename, index=False)


# Get all the txt files in the folder
folder_path = 'File Path'  # txt file folder path
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# Use multiprocess technology to process all txt files
if __name__ == '__main__':
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_file, [os.path.join(folder_path, f) for f in txt_files])



