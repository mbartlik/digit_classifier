import torch
import torchvision
import torch.nn as nn
from model_class import Model

checkpoint = torch.load('models/model_20')
print(checkpoint['hidden_layers_count'])
print(checkpoint['hidden_layer_size'])

hidden_layers_count = checkpoint['hidden_layers_count']
hidden_layer_size = checkpoint['hidden_layer_size']
dropout = 0.1


model = Model(hidden_layers_count, hidden_layer_size, 0.1)

model.load_state_dict(checkpoint['state_dict'])


transform = torchvision.transforms.Compose([
											torchvision.transforms.ToTensor(),
											torchvision.transforms.Normalize((0.1307,),(0.3081,))
											])

test_dataset = torchvision.datasets.MNIST('MNIST_dataset', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

running_accuracy = 0

model.eval()
for images, labels in test_loader:

	images = images.reshape(images.shape[0],-1)
	logits = model(images, hidden_layers_count)
	ps, predictions = torch.topk(logits,1,dim=1)
	
	correct = 0
	for i in range(len(labels)):
		if labels[i] == predictions[i]:
			correct += 1

	running_accuracy += float(correct)/len(labels)

print("Accuracy: {}".format(running_accuracy/len(test_loader)))







