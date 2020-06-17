import torch


"""
This file generates a report on details of each model listed in the filenames section
"""


# List of model filepaths to report on
filenames = ['models/model_1', 'models/model_2', 'models/model_3', 'models/model_4', 'models/model_5', 'models/model_6', 'models/model_7', 'models/model_8', 'models/model_9', 'models/model_10', 'models/model_11', 'models/model_12', 'models/model_13', 'models/model_14', 'models/model_16', 'models/model_17', 'models/model_18', 'models/model_19', 'models/model_20', 'models/model_21'] 

# Placeholder for the best model - this will be changed later
best_model = torch.load(filenames[0])
best_model_name = filenames[0]

# Loop through filepaths
for filename in filenames:

	# Load checkpoint and print info about model
	checkpoint = torch.load(filename)
	print("Model: {}".format(filename))
	print("Learn rate: {}".format(checkpoint['learn_rate']))
	print("Epochs: {}".format(checkpoint['epochs']))
	print("Hidden layers: {}".format(checkpoint['hidden_layers_count']))
	print("Hidden layer size: {}".format(checkpoint['hidden_layer_size']))
	print("Accuracy: {}".format(checkpoint['accuracy']))
	print("--------")

	# If the current model is more accurate than the best model then update the best model
	if checkpoint['accuracy'] > best_model['accuracy']:
		best_model = checkpoint
		best_model_name = filename

# Output the best model name and accuracy
print("The best model is {}".format(best_model_name))
print("It's accuracy is {0:.2f}%".format(best_model['accuracy']*100))