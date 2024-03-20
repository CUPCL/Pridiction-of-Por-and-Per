import pandas as pd
import multiprocessing as mp


# Load the b table and convert it into a dictionary
def load_b_table(b_file):
    b_table = pd.read_excel(b_file)
    b_dict = {}
    for index, row in b_table.iterrows():
        well_name = row['Well name']
        depth = row['Depth']
        other_features = row.drop(['Well name', 'Depth']).to_dict()

        if well_name not in b_dict:
            b_dict[well_name] = []
        b_dict[well_name].append({'Well name': well_name, 'Depth': depth, 'feature': other_features})
    return b_dict


# Process a row in table A
def process_a_row(args):
    row, b_dict = args
    well_name = row['Well name']
    depth = row['Depth']
    other_data = row.drop(['Well name', 'Depth']).to_dict()

    # Initializes the matching feature as an empty dictionary and associated variables
    matched_features = {}
    closest_depth_diff = float('inf')  # Initialize an infinite depth difference
    closest_b_row = None  # Used to store table B rows with the closest depth

    # If the well name is in b_dict, iterate through the corresponding data to find a match
    if well_name in b_dict:
        b_well_data = sorted(b_dict[well_name], key=lambda x: x['Depth'])  # Sort by depth in ascending order

        # Walk through all the data for the well in table B
        for current_b_row in b_well_data:
            current_b_depth = current_b_row['Depth']
            depth_diff = abs(depth - current_b_depth)  # Calculated depth difference

            # Check for a smaller depth difference
            if depth_diff < closest_depth_diff:
                closest_depth_diff = depth_diff
                matched_features = current_b_row.get('feature', {})  # Update matching feature
                closest_b_row = current_b_row

                # If no exact match is found, the closest one (the one with the smallest depth difference) is used.

    # Merge result dictionary
    result = {'Well name': well_name, 'Depth': depth, **other_data, **matched_features}
    return result


# Use multiple processes to process the a table
def process_a_table(a_file, b_dict, output_file, num_processes=mp.cpu_count()):
    a_table = pd.read_excel(a_file)

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_a_row, [(row, b_dict) for index, row in a_table.iterrows()])

    c_table = pd.DataFrame(results)
    c_table.to_excel(output_file, index=False)


# Main function
def main():
    b_file = "File Path"  # Notice: This should be table B, load it into b_dict
    a_file = "File Path"  # Notice：This should be table A, which should be used as the baseline for data matching
    output_file = "File Path"

    b_dict = load_b_table(b_file)
    process_a_table(a_file, b_dict, output_file)

if __name__ == "__main__":
    main()
