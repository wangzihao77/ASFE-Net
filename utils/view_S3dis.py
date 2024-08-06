import numpy as np
import glob, os
from helper_ply import read_ply, write_ply

# Define the color mapping function
def create_color_map(num_classes):
    np.random.seed(0)  # To ensure consistent color, use fixed random seeds
    return np.random.randint(0, 256, (num_classes, 3), dtype=np.uint8)

# Coloring function
def create_colored_ply(points, labels, filename, num_classes, colors):
    # Color each point according to the label
    point_colors = colors[labels]
    # Merges point coordinates and colors into an array ready to be written
    field_list = [points, point_colors]
    field_names = ['x', 'y', 'z', 'red', 'green', 'blue']
    # Call the write_ply function to write to the file
    write_ply(filename, field_list, field_names)

if __name__ == '__main__':
    # Path setting
    base_dir = '/Users/apple/Desktop/randla-net-tf2-update/ply'
    original_data_dir = '/Users/apple/Desktop/randla-net-tf2-update/orinal'
    num_classes = 13

    colors = create_color_map(num_classes)


    data_path = glob.glob(os.path.join(base_dir, '*.ply'))
    data_path = np.sort(data_path)

    for file_name in data_path:

        pred_data = read_ply(file_name)
        pred_labels = pred_data['pred']  #
        original_data = read_ply(os.path.join(original_data_dir, os.path.basename(file_name)[:-4] + '.ply'))
        original_labels = original_data['class']
        points = np.vstack((original_data['x'], original_data['y'], original_data['z'])).T


        output_file_pred = os.path.join(base_dir, os.path.basename(file_name)[:-4] + '_pred_colored.ply')
        create_colored_ply(points, pred_labels, output_file_pred, num_classes, colors)


        output_file_orig = os.path.join(original_data_dir, os.path.basename(file_name)[:-4] + '_orig_colored.ply')
        create_colored_ply(points, original_labels, output_file_orig, num_classes, colors)

        print(f'Finished creating colored PLY files for: {file_name}')
