import numpy as np
import glob
import os
from helper_ply import read_ply, write_ply


def create_color_map(num_classes):
    np.random.seed(42)
    return np.random.randint(0, 256, (num_classes, 3), dtype=np.uint8)


def colorize_and_save_ply(points, labels, color_map, original_dir, base_filename, postfix):
    point_colors = color_map[labels]
    colored_points = np.column_stack((points, point_colors))


    new_file_name = f"{base_filename}_{postfix}.ply"
    new_file_path = os.path.join(original_dir, new_file_name)


    write_ply(new_file_path, colored_points, ['x', 'y', 'z', 'red', 'green', 'blue'])
    print(f"Saved: {new_file_path}")


if __name__ == '__main__':

    num_classes = 9

    color_map = create_color_map(num_classes)


    original_dir = '/Users/apple/Desktop/visualize_toronto/toronto_orinal'
    prediction_dirs = [
        '/Users/apple/Desktop/visualize_toronto/toronto_ours',
        '/Users/apple/Desktop/visualize_toronto/toronto_randla'
    ]


    original_ply_files = glob.glob(os.path.join(original_dir, '*.ply'))


    for original_ply_path in original_ply_files:
        original_ply_data = read_ply(original_ply_path)
        original_points = np.vstack((original_ply_data['x'], original_ply_data['y'], original_ply_data['z'])).T
        original_labels = original_ply_data['scalar_Label']


        if not issubclass(original_labels.dtype.type, np.integer):
            original_labels = original_labels.astype(int)


        base_filename = os.path.splitext(os.path.basename(original_ply_path))[0]


        colorize_and_save_ply(original_points, original_labels, color_map, original_dir, base_filename, 'original_colored')


        for prediction_dir in prediction_dirs:
            prediction_ply_path = os.path.join(prediction_dir, os.path.basename(original_ply_path))
            if os.path.exists(prediction_ply_path):
                prediction_ply_data = read_ply(prediction_ply_path)
                prediction_points = np.vstack((prediction_ply_data['x'], prediction_ply_data['y'], prediction_ply_data['z'])).T
                prediction_labels = prediction_ply_data['preds']


                assert prediction_points.shape == original_points.shape, "Point clouds are not aligned."


                postfix = os.path.basename(prediction_dir).split('/')[-1]
                colorize_and_save_ply(prediction_points, prediction_labels, color_map, original_dir, base_filename, f'prediction_{postfix}_colored')

    print("Colorized PLY files have been saved next to the original PLY files.")
