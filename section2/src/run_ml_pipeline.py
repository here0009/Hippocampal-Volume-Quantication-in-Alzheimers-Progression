"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import random
from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

random.seed(21)


class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = os.path.abspath(os.path.join(*['..', '..', 'section1', 'out']))
        self.n_epochs = 2
        self.learning_rate = 0.0002
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = os.path.abspath(os.path.join(*['..', 'out']))

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it.
    data = LoadHippocampusData(c.root_dir, y_shape=c.patch_size, z_shape=c.patch_size)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))
    print(keys)
    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation
    # and testing respectively.
    # <YOUR CODE GOES HERE>
    train_weights = [0.8, 0.1, 0.1]
    assert sum(train_weights) == 1
    train_len, val_len, _ = [int(_r * len(data)) for _r in train_weights]
    indices = list(keys)
    random.shuffle(indices)
    split['train'] = indices[:train_len]
    split['val'] = indices[train_len: train_len + val_len]
    split['test'] = indices[train_len + val_len:]
    print(f'The length of train, validation and test dataset is {train_len}, {val_len}, {len(indices) - train_len - val_len}')
    # Set up and run experiment
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    del data

    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

