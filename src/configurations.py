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
    target_col_name = output_classes = dataset_info = None

    if dataset_name == 'Artif-1':
        output_classes = (OutputClass(name='y0', encoding=0),
                          OutputClass(name='y1', encoding=1))
        dataset_info = DatasetMetaData(name=dataset_name, target_col='y', output_classes=output_classes, n_inputs=5,
                                       n_outputs=2)
    elif dataset_name == 'Artif-2':
        output_classes = (OutputClass(name='y0', encoding=0),
                          OutputClass(name='y1', encoding=1))
        dataset_info = DatasetMetaData(name=dataset_name, target_col='y', output_classes=output_classes, n_inputs=5,
                                       n_outputs=2)
    elif dataset_name == 'MB-PAM50':
        output_classes = (OutputClass(name='lumA', encoding=0),
                          OutputClass(name='lumB', encoding=1),
                          OutputClass(name='lumC', encoding=2),
                          OutputClass(name='lumD', encoding=3),
                          OutputClass(name='lumE', encoding=4))
        dataset_info = DatasetMetaData(name=dataset_name, target_col='PAM50', output_classes=output_classes, n_inputs=1350,
                                       n_outputs=5)
    # elif dataset_name == 'MB-GE-ER':
    #     output_classes = (OutputClass(name='-', encoding=0),
    #                       OutputClass(name='+', encoding=1))
    #     dataset_info = DatasetMetaData(name=dataset_name, target_col='ER_Expr', output_classes=output_classes,
    #                                    n_inputs=1000, n_outputs=2)
    # elif dataset_name == 'BreastCancer':
    #     output_classes = (OutputClass(name='M', encoding=0),
    #                       OutputClass(name='B', encoding=1))
    #     dataset_info = DatasetMetaData(name=dataset_name, target_col='diagnosis', output_classes=output_classes,
    #                                    n_inputs=30, n_outputs=2)
    # elif dataset_name == 'LetterRecognition':
    #     output_classes = (OutputClass(name='A', encoding=0),
    #                       OutputClass(name='B-Z', encoding=1))
    #     dataset_info = DatasetMetaData(name=dataset_name, target_col='letter', output_classes=output_classes,
    #                                    n_inputs=16, n_outputs=2)
    # elif dataset_name == 'MNIST':
    #     output_classes = (OutputClass(name='0', encoding=0),
    #                       OutputClass(name='1-9', encoding=1))
    #     dataset_info = DatasetMetaData(name=dataset_name, target_col='digit', output_classes=output_classes,
    #                                    n_inputs=784, n_outputs=2)
    else:
        print('WARNING: invalid dataset name given!')

    return dataset_info

# TODO: datasets to add
# Multiclass (MNIST, LetterRecognition
