
from matplotlib import pyplot as plt
import os
import time

import numpy as np
import scipy as sp
import pandas as pd
import random as rd


def make_kernel_plot():
    from function_file import set_kernel
    kernel = set_kernel((1, 3))

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


def combine_csv_files(output_filename='combined_data', to_csv=True, excel=False):
    folder_path = 'csv_files/'
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
                subset = components[8][-1] == 'T'
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


def plot_iou_comparison():
    # Load CSV file into a DataFrame
    data = pd.read_csv('combined_data.csv')

    # Filter data for entries with 50 epochs
    filtered_data = data[data['Epochs'] == 50]

    # Extract relevant columns for plotting
    model_names = filtered_data['Name']
    iou5_scores = filtered_data['IoU5']
    val_iou5_scores = filtered_data['val_IoU5']

    # Plot the data
    plt.scatter(iou5_scores, val_iou5_scores, marker='o', color='b')
    plt.title('IoU5 vs val_IoU5 Comparison at 50 Epochs')
    plt.xlabel('IoU5')
    plt.ylabel('val_IoU5')

    # Annotate each point with the corresponding model name
    for i, model_name in enumerate(model_names):
        plt.annotate(model_name, (iou5_scores.iloc[i], val_iou5_scores.iloc[i]))

    plt.grid(True)
    plt.show()


df = pd.read_csv('combined_data.csv')
plot_iou_comparison()

if __name__ == '__ma in__':
    combine_csv_files()
