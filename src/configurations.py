# Configurations for datasets
# TODO do this using proper configs later
from collections import namedtuple

from rules.rule import OutputClass
DatasetMetaData = namedtuple('DatasetMetaData', 'name target_col output_classes n_inputs n_outputs')


def get_configuration(dataset_name):
    """
    Return target_col_name and output class encodings for dataset accordingly
    i.e. Class names with their corresponding encoding (output neuron index)
    """
    target_col_name = output_classes = None

    # if dataset_name == 'Artif-1':
    #     target_col_name = 'y'
    #     output_classes = (OutputClass(name='y0', encoding=0),
    #                       OutputClass(name='y1', encoding=1))

    if dataset_name == 'Artif-2':
        output_classes = (OutputClass(name='y0', encoding=0),
                          OutputClass(name='y1', encoding=1))
        dataset_info = DatasetMetaData(name='Artif-2', target_col='y', output_classes=output_classes, n_inputs=5,
                                       n_outputs=2)
    elif dataset_name == 'MB-GE-ER':
        output_classes = (OutputClass(name='-', encoding=0),
                          OutputClass(name='+', encoding=1))
        dataset_info = DatasetMetaData(name='MB-GE-ER', target_col='ER_Expr', output_classes=output_classes, n_inputs=1000,
                                       n_outputs=2)
    else:
        print('WARNING: invalid dataset name given!')

    # elif dataset_name == 'BreastCancer':
    #     target_col_name = 'diagnosis'
    #     output_classes = (OutputClass(name='M', encoding=0),
    #                       OutputClass(name='B', encoding=1))

    return dataset_info
