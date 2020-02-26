# Configurations for datasets
# TODO do this using proper configs later

from rules.rule import OutputClass


def get_configuration(dataset_name):
    """
    Return target_col_name and output class encodings for dataset accordingly
    i.e. Class names with their corresponding encoding (output neuron index)
    """
    target_col_name = output_classes = None

    if dataset_name == 'Artif-1':
        target_col_name = 'y'
        output_classes = (OutputClass(name='y0', encoding=0),
                          OutputClass(name='y1', encoding=1))

    elif dataset_name == 'Artif-2':
        target_col_name = 'y'
        output_classes = (OutputClass(name='y0', encoding=0),
                          OutputClass(name='y1', encoding=1))

    elif dataset_name == 'BreastCancer':
        target_col_name = 'diagnosis'
        output_classes = (OutputClass(name='M', encoding=0),
                          OutputClass(name='B', encoding=1))

    return target_col_name, output_classes
