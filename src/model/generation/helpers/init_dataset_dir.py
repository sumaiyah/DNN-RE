"""
Initialise the empty dataset directory

`data.csv`
`neural_network_initialisation/`
    `best_initialisation.h5`
    `data_split_indices.txt`
`results/`
    `grid_search_results.txt`
    `labels.csv`
    `rule_ex_results.csv`
    TODO: `saved_rules`
`cross_validation/`
    `<n>_fold/`
        `data_split_indices.txt`
        `trained_models/`
"""
import os


def create_directory(dir_path):
    """

    Args:
        dir_path: path to the new directory
    """
    # Create directory given path if it doesnt exist
    try:
        os.makedirs(dir_path)
        print("Directory ", dir_path, " Created ")
    except FileExistsError:
        print("Directory ", dir_path, " already exists")


def run(dataset_name, path_to_data_folder):
    """

    Args:
        dataset_name: e.g. 'MB-GE-ER' or 'MNIST'
        path_to_data_folder: path to main data/ folder for project

    Creates empty dataset directory as specified above
    """
    # Base directory
    base_path = path_to_data_folder + dataset_name + '/'
    create_directory(base_path)

    # neural_network_initialisation/
    create_directory(dir_path=base_path + 'neural_network_initialisation')

    # results
    create_directory(dir_path=base_path + 'results')

    # cross_validation/
    create_directory(dir_path=base_path + 'cross_validation')


def clean_up():
    from src import TEMP_DIR
    def clear_dir(dir_path):
        for file in os.listdir(dir_path):
            if os.path.isdir(dir_path + file):
                clear_dir(dir_path + file + '/')
            else:
                os.remove(dir_path + file)

    # Remove temporary files at the end
    print('Cleaning up temporary files...', end='', flush=True)
    clear_dir(TEMP_DIR)
    print('done')
