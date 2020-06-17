import torch
from digit_classifier_generator import digit_classifier_generator


"""
This file runs the digit_classifier_generator on various combinations of hyperparameters
"""


# Hyperparameters to be tried
learn_rates = [0.001]
epochs = [20]
hidden_layers_counts = [2]
hidden_layer_sizes = [128]
rand_seeds = [23,50]
dropouts = [0.1,0.2,0.3]

# Save models under these filepaths
filenames = ['models/model_22', 'models/model_23', 'models/model_24', 'models/model_25', 'models/model_26', 'models/model_27']
file_number = 0

# Loop through so each hyperparameter combination is tried
for learn_rate in learn_rates:
	for epoch_count in epochs:
		for hidden_layers_count in hidden_layers_counts:
			for hidden_layer_size in hidden_layer_sizes:
				for rand_seed in rand_seeds:
					for dropout in dropouts:
						digit_classifier_generator(filenames[file_number],learn_rate=learn_rate, epochs=epoch_count, hidden_layers_count=hidden_layers_count, hidden_layer_size=hidden_layer_size, rand_seed=rand_seed, dropout=dropout)
						print("Saving {}".format(filenames[file_number]))
						print("----------------------")
						file_number += 1




"""
Completed hyperparameters attempts

1st try:
learn_rates = [0.001,0.005,0.01]
epochs = [10,15]
hidden_layers_counts = [2,3]
*** Notes: keep learn rate low, epochs high, hidden_layers_counts can be 2 or 3
-----------------

2nd try:
learn_rates = [0.0005,0.001]
epochs = [20]
hidden_layers_counts = [2]
hidden_layer_size = [64,128]
-----------------

3rd try:
learn_rates = [0.001]
epochs = [20]
hidden_layers_counts = [2]
hidden_layer_sizes = [128]
rand_seeds = [4,8,15,16,23,42]
-----------------

4th try:
learn_rates = [0.001]
epochs = [20]
hidden_layers_counts = [2]
hidden_layer_sizes = [128]
rand_seeds = [23,50]
dropouts = [0.1,0.2,0.3]
-----------------
"""

