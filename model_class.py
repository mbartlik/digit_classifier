import torch.nn as nn

class Model(nn.Module):
		def __init__(self, hidden_layers_count, hidden_layer_size, dropout):
			super().__init__()

			self.h1 = nn.Linear(784,hidden_layer_size)

			# Add more hidden layers depending on hidden_layer_count
			if hidden_layers_count > 1:
				self.h2 = nn.Linear(hidden_layer_size,hidden_layer_size)

			if hidden_layers_count > 2:
				self.h3 = nn.Linear(hidden_layer_size,hidden_layer_size)

			if hidden_layers_count > 3:
				self.h4 = nn.Linear(hidden_layer_size,hidden_layer_size)

			self.out = nn.Linear(hidden_layer_size,10)

			self.ReLU = nn.ReLU()

			self.dropout = nn.Dropout(p=dropout)
		# Define forward pass
		def forward(self, x, hidden_layers_count):

			x = self.dropout(x)
			x = self.h1(x)
			x = self.ReLU(x)
			
			if hidden_layers_count > 1:
				x = self.dropout(x)
				x = self.h2(x)
				x = self.ReLU(x)
			if hidden_layers_count > 2:
				x = self.dropout(x)
				x = self.h3(x)
				x = self.ReLU(x)
			if hidden_layers_count > 3:
				x = self.Dropout(x)
				x = self.h4(x)
				x = self.ReLU(x)

			x = self.out(x)

			return x