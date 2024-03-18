from matplotlib import pyplot as plt
import os
import time

import numpy as np
import scipy as sp
import pandas as pd


def make_kernel_plot():
    from function_file import set_kernel
    kernel = set_kernel((2, 3))

    # Create a meshgrid for x, y coordinates
    x = np.arange(kernel.shape[0])
    y = np.arange(kernel.shape[1])
    x, y = np.meshgrid(x, y)

    # Flatten the matrices for bar3d function
    x = x.flatten()
    y = y.flatten()
    z = np.zeros_like(x)
    dx = dy = 1
    dz = kernel.flatten()

    # Plot the 3D bar plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x, y, z, dx, dy, dz, shade=True)

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Kernel Values')

    # Show the plot
    plt.show()


def combine_csv_files(sub_folder, output_filename='combined_data', to_csv=True, excel=False):
    folder_path = f'csv_files/{sub_folder}'
    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            # Add a new column with the folder name as an identifier
            components = filename.split('_')

            # Extract relevant information from the components
            size = components[2]

            val_present = 'v' in components[3]
            if val_present:
                val_size = components[3][1:]
                rotate = components[4][-1] == 'T'
                new_background = components[5][-1] == 'T'
                val_rotate = components[6][-1] == 'T'
                val_new_background = components[7][-1] == 'T'
                subset = components[8][-5] == 'T'
            else:
                rotate = components[3][-1] == 'T'
                new_background = components[4][-1] == 'T'
                val_size = None
                val_rotate = None
                val_new_background = None
                subset = None

            df.insert(0, 'Name', components[0])
            df['LineSize'] = size
            df['Rotate'] = rotate
            df['NewBackground'] = new_background

            df['ValLineSize'] = val_size
            df['ValRotate'] = val_rotate
            df['ValNewBackground'] = val_new_background
            df['Subset'] = subset

            # Concatenate the data to the combined DataFrame
            combined_data = pd.concat([combined_data, df], ignore_index=True)

    # Save the combined data to a new CSV file
    combined_data.rename(columns={'Unnamed: 0': 'Epochs'}, inplace=True)
    combined_data['Epochs'] = combined_data['Epochs'] + 1

    if to_csv:
        combined_data.to_csv(output_filename + '.csv', index=False)
    if excel:
        combined_data.to_excel(output_filename + '.xlsx', index=False)


def plot_comparison(data_path, metrics, epochs):
    num_epoch = epochs
    # Load CSV file into a DataFrame
    data = pd.read_csv(data_path)
    print(data.columns)

    # Filter data for entries with 10 epochs, Rotate is True, and Subset is False
    filtered_data = data[(data['Epochs'] == num_epoch) & (data['Rotate'] == True) & (data['Subset'] == False)]
    filtered_data['Train_time'] = filtered_data['Train_time'] / 60  # / 60  # Convert Train_time to hours

    # Extract relevant columns for plotting
    model_names = filtered_data['Name']

    # Plot the data
    fig, ax = plt.subplots()

    num_metrics = len(metrics)
    bar_width = 0.2

    # Set positions for the bars
    bar_positions = [range(len(model_names))]
    for i in range(1, num_metrics):
        bar_positions.append([pos + i * bar_width for pos in bar_positions[0]])

    # Iterate over metrics and plot bars
    for i, metric in enumerate(metrics):
        ax.bar(bar_positions[i], filtered_data[metric], width=bar_width, label=metric)

    # Set labels and title
    ax.set_xlabel('Name')
    ax.set_ylabel('Metrics')
    ax.set_title(f'Comparison of Metrics at {str(num_epoch)} Epochs')
    ax.set_xticks([pos + (num_metrics - 1) * bar_width / 2 for pos in bar_positions[0]])
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.grid(True)
    plt.show()


def train_time_print(time_start):
    time_end = time.time() - time_start

    hours, remainder = divmod(int(time_end), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        print(f"Training time: {hours} hours, {minutes} minutes, and {seconds} seconds")
    elif minutes > 0:
        print(f"Training time: {minutes} minutes and {seconds} seconds")
    else:
        print(f"Training time: {seconds} seconds")


make_kernel_plot()

if __name__ == '__ma in__':
    # make_rec_weights(100, 7, False, True)
    combine_csv_files('short-box')
    metrics_to_compare = ['loss','IoU9', 'Train_time']
    plot_comparison('combined_data.csv', metrics_to_compare, 5)
