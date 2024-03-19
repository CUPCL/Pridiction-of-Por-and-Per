import pandas as pd
import os
import chardet
from multiprocessing import Pool


def process_file(file_path):
    # 检测文件的编码
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

        # 读取txt文件
    data = pd.read_csv(file_path, sep='\s+', header=0, encoding=encoding)  # 使用检测到的编码

    # 获取文件名（去除.txt）作为Excel文件名和新的表头
    filename = os.path.basename(file_path).replace('.txt', '')
    data['井名'] = filename  # 将文件名添加为新的列

    # 对数据进行重新排序，确保“井名”列为第一列
    data = data[['井名'] + [col for col in data.columns if col != '井名']]

    # 获取txt文件的目录路径，并将其作为Excel文件的保存路径
    directory = os.path.dirname(file_path)
    excel_filename = os.path.join(directory, filename + '.xlsx')
    data.to_excel(excel_filename, index=False)


# 获取文件夹内所有的txt文件
folder_path = 'File Path'  # txt文档文件夹路径
txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

# 使用多进程技术处理所有txt文件
if __name__ == '__main__':
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_file, [os.path.join(folder_path, f) for f in txt_files])



