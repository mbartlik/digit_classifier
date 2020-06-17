import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from model_class import Model



def digit_classifier_generator(filepath, train_batch_size=64, learn_rate=0.005, epochs=10, optimizer_type='Adam', dropout=0.15, hidden_layers_count=2, hidden_layer_size=128, rand_seed=7):
	
	# Hyperparameters
	train_batch_size = train_batch_size
	learn_rate = learn_rate
	epochs = epochs
	optimizer_type = optimizer_type
	dropout = dropout
	hidden_layers_count = hidden_layers_count
	hidden_layer_size = hidden_layer_size
	rand_seed = rand_seed

	# Set random seed
	torch.manual_seed(rand_seed)


	# Define transform
	transform = torchvision.transforms.Compose([
											torchvision.transforms.ToTensor(),
											torchvision.transforms.Normalize((0.1307,),(0.3081,))
											])

	# Load datasets
	train_dataset = torchvision.datasets.MNIST('MNIST_dataset', train=True, transform=transform)
	test_dataset= torchvision.datasets.MNIST('MNIST_dataset', train=False, transform=transform)

	# Make data loaders
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)




	# Create model object
	model = Model(hidden_layers_count, hidden_layer_size, dropout)

	# Make an optimizer
	if optimizer_type == 'SGD':
		optimizer = optim.SGD(model.parameters(), lr=learn_rate)
	elif optimizer_type == 'Adam':
		optimizer = optim.Adam(model.parameters(), lr=learn_rate)

	# Define the loss
	criterion = nn.CrossEntropyLoss()

	for e in range(epochs):

		training_loss = 0

		# Turns on dropout
		model.train()

		# Decay learning rate
		if e == 10:
			optimizer.lr = learn_rate/2
			print("Decaying learn rate")
		if e == 15:
			optimizer.lr = learn_rate/2
			print("Decaying learn rate")

		# Training stage
		for images, labels in train_loader:

			# Flatten images - will be 64x784
			images = images.reshape(images.shape[0],-1)

			# Zero out optimizer
			optimizer.zero_grad()
		
			# Forward pass through model - results in un-normalized logits
			logits = model(images, hidden_layers_count)

			# Calculate loss and add to training loss
			loss = criterion(logits,labels)
			training_loss += loss

			# Find gradient then optimize
			loss.backward()
			optimizer.step()

		print("Training loss {}".format(training_loss/len(train_loader)))

		with torch.no_grad():

			# Turn off dropout for evaluation
			model.eval()

			running_accuracy = 0
			for images, labels in test_loader:

				# Flatten images
				images = images.reshape(images.shape[0],-1)

				# Find predictions of the model
				logits = model(images, hidden_layers_count)
				probs, predictions = torch.topk(logits,1,dim=1)

				# See where the predictions match up with the labels
				correct_points = predictions == labels.view(*predictions.shape)

				# Convert booleans to 1's and 0's to find accuracy
				correct_points = correct_points.type(torch.FloatTensor)

				# Calculate accuracy
				accuracy = torch.sum(correct_points)/len(correct_points)

				# Add to total
				running_accuracy += accuracy.item()

			running_accuracy = running_accuracy/len(test_loader)

			print("Testing accuracy: {}".format(running_accuracy))




	# Save model
	checkpoint = {'hidden_layer_size': hidden_layer_size,
				  'hidden_layers_count': hidden_layers_count,
				  'learn_rate': learn_rate,
				  'epochs': epochs,
				  'optimizer_type': optimizer_type,
				  'dropout': dropout,
				  'accuracy': running_accuracy,
				  'dropout': dropout,
				  'state_dict': model.state_dict()}

	torch.save(checkpoint,filepath)
					


