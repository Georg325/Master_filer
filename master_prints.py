from matplotlib import pyplot as plt
import os
import time

import numpy as np
import pandas as pd
import re
from function_file import make_folder


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
            df = pd.read_csv(file_path).tail(1)
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
        combined_data.to_csv(sub_folder + '_' + output_filename + '.csv', index=False)
    if excel:
        combined_data.to_excel(sub_folder + '_' + output_filename + '.xlsx', index=False)


def plot_comparison(sub_folder, metrics, data_path='combined_data.csv', sort_by=None):
    # Load CSV file into a DataFrame
    data = pd.read_csv(sub_folder + '_' + data_path)
    if sort_by is None:
        sort_by = metrics[0]
    data.sort_values(by=[sort_by], inplace=True, ascending=False)

    # Filter data for entries with 10 epochs, Rotate is True, and Subset is False
    filtered_data = data
    filtered_data['Train_time'] = filtered_data['Train_time'] / 60 / 60  # Convert Train_time to hours

    # Extract relevant columns for plotting
    model_names = filtered_data['Name']

    # Plot the data
    fig, ax = plt.subplots()

    num_metrics = len(metrics)
    bar_width = 0.35

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
    ax.set_title(f'Comparison of Metrics for {sub_folder}')
    ax.set_xticks([pos + (num_metrics - 1) * bar_width / 2 for pos in bar_positions[0]])
    ax.set_xticklabels(model_names)
    ax.legend()

    plt.grid(True)

    plt.show()


def parse_plots(sub_folder):
    folder_path = f'csv_files/{sub_folder}'
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = f'{folder_path}/{filename}'
            ind_plot(file_path, sub_folder)


def ind_plot(filepath='csv_files/tul/dense_e5_(6, 2)_v(2, 6)_rF_bT_rvF_bvT_subF.csv', sub_folder=''):
    df = pd.read_csv(filepath, index_col=0)
    df.pop('Train_time')

    epoch = int(re.search(r'_e(\d+)_', filepath).group(1))

    list_of_metrics = df.columns

    metric_groups = [[] for _ in range(6)]
    for metric in list_of_metrics:
        if 'val_' in metric:
            if 'loss' in metric:
                metric_groups[0].append(metric)
            elif 'ec' in metric:
                metric_groups[1].append(metric)
            else:
                metric_groups[2].append(metric)
        else:
            if 'loss' in metric:
                metric_groups[3].append(metric)
            elif 'ec' in metric:
                metric_groups[4].append(metric)
            else:
                metric_groups[5].append(metric)

    if len(metric_groups[0]) > 0:
        fig, ax = plt.subplots(2, 2, figsize=(11, 11))
        plot_metrics = ([metric_groups[0], metric_groups[3]], [metric_groups[1], metric_groups[4]], [metric_groups[5]],
                        [metric_groups[2]])
        titles = ['Loss', 'Precision and Recall', 'IoU', 'alt_IoU']

        for k, metric in enumerate(plot_metrics):
            for i in range(len(metric[0])):

                if len(metric) == 2:
                    ax[k // 2, k % 2].plot(np.array(df[metric[0][i]]), label=metric[0][i])
                    ax[k // 2, k % 2].plot(np.array(df[metric[-1][i]]), label=metric[-1][i])
                else:
                    ax[k // 2, k % 2].plot(np.array(df[metric[0][i]]), label=f'Step {i + 1}')

                ax[k // 2, k % 2].set_ylim(0, 1)
                ax[k // 2, k % 2].set_xlabel('Epoch')
                if epoch >= 29:
                    ax[k // 2, k % 2].set_xlim(0, epoch + 1)
                else:
                    ax[k // 2, k % 2].set_xlim(0, epoch)
                ax[k // 2, k % 2].set_title(titles[k])
                ax[k // 2, k % 2].grid('on')
                if titles[k] != 'IoU':
                    ax[k // 2, k % 2].legend()
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        plot_metrics = (metric_groups[3], metric_groups[4], metric_groups[5])
        titles = ['Loss', 'Precision and Recall', 'IoU']

        for k, metric in enumerate(plot_metrics):
            for i in range(len(metric)):
                if titles[k] == 'IoU':
                    ax[k].plot(np.array(df[metric[i]]), label=f'Step {i + 1}')
                else:
                    ax[k].plot(np.array(df[metric[i]]), label=metric[i])
                ax[k].set_ylim(0, 1)
                if epoch >= 29:
                    ax[k].set_xlim(0, epoch + 1)
                else:
                    ax[k].set_xlim(0, epoch)
                ax[k].set_xlabel('Epoch')
                ax[k].set_title(titles[k])
                ax[k].grid('on')
                ax[k].legend()
    name_eat = filepath.split('/')[-1].split('_')[0]
    fig.suptitle(f'Metrics from the ' + name_eat + ' model ' + sub_folder)
    fig.tight_layout()
    make_folder(sub_folder)
    plt.savefig(sub_folder + '/' + filepath.split('/')[-1].split('.')[0] + '.pdf')


# make_kernel_plot()
# parse_plots('box')

# ind_plot('csv_files/tul/dense_e100_(6, 2)_v(2, 6)_rF_bT_rvF_bvT_subF.csv')

if __name__ == '__main__':
    # make_rec_weights(100, 7, False, True)
    combine_csv_files('box')
    metrics_to_compare = ['loss', 'val_loss']
    plot_comparison('box', metrics_to_compare)
    metrics_to_compare = ['precision', 'recall']
