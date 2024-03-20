import pandas as pd
import multiprocessing

'''This code completes pairwise merging and feature alignment of the excel file'''

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df


if __name__ == '__main__':
    # Read two excel tables
    file1 = 'File Path'
    file2 = 'File Path'

    # Read Excel files using parallel processing
    pool = multiprocessing.Pool()
    df1 = pool.apply_async(read_excel_file, args=(file1,))
    df2 = pool.apply_async(read_excel_file, args=(file2,))
    pool.close()
    pool.join()

    # Merge data from excel file
    merged_data = pd.concat([df1.get(), df2.get()], ignore_index=True)

    # Save the results to a new excel file
    merged_data.to_excel('File Path', index=False)