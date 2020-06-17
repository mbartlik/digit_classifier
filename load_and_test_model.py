import torch
import torchvision
import torch.nn as nn
from model_class import Model


def load_and_test_model(filepath):

	"""This function loads and tests accuracy of a model
	Input: filepath to the desired model
	"""

	# Print the model that is to be loaded
	print("Model of filepath {}".format(filepath))
	
	# Load the desired checkpoint
	checkpoint = torch.load(filepath)

	# Attain model structure info
	hidden_layers_count = checkpoint['hidden_layers_count']
	hidden_layer_size = checkpoint['hidden_layer_size']

	# Create model object (with random parameters for now)
	model = Model(hidden_layers_count, hidden_layer_size, 0.1)

	# Load state_dict into model from checkpoint
	model.load_state_dict(checkpoint['state_dict'])


	# Define transform for the testing dataset
	transform = torchvision.transforms.Compose([
												torchvision.transforms.ToTensor(),
												torchvision.transforms.Normalize((0.1307,),(0.3081,))
												])
	# Load test data
	test_dataset = torchvision.datasets.MNIST('MNIST_dataset', train=False, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)


	running_accuracy = 0

	# Turns off dropout
	model.eval()

	# Loop through images and labels in test_loader
	for images, labels in test_loader:

		# Flatten images
		images = images.reshape(images.shape[0],-1)

		# Obtain logits
		logits = model(images, hidden_layers_count)

		# Obtain the index with highest logit in each row (this is the prediction)
		ps, predictions = torch.topk(logits,1,dim=1)
		
		# Compare the labels to predictions and count correct predictions
		correct = 0
		for i in range(len(labels)):
			if labels[i] == predictions[i]:
				correct += 1

		# Add the accuracy for this batch to the running accuracy		
		running_accuracy += float(correct)/len(labels)

	# The accuracy is the running acuracy divided by amount of batches in the test data
	print("Accuracy: {0:.2f}%".format(running_accuracy*100/len(test_loader)))


load_and_test_model('models/model_20')




