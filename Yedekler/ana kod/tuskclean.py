import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))

        for i in range(len(hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i + 1]))
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))

        # Output katmanı
        self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, inputs):
        layers = [inputs]
        for i in range(len(self.weights)):
            layers.append(np.dot(layers[-1], self.weights[i]) + self.biases[i])
            if i == len(self.weights) - 1:
                finalOutput = np.exp(layers[-1] - np.max(layers[-1], axis=1, keepdims=True))
                finalOutput = finalOutput / np.sum(finalOutput, axis=1, keepdims=True)
                layers.append(finalOutput)
            else:
                layers.append(np.maximum(0, layers[-1]))
        return layers[-1]

    @staticmethod
    def LossCategoricalCrossEntropy(yPred, yTrue):
        yPred = np.clip(yPred, 1e-10, 1 - 1e-10)
        loss = -np.sum(yTrue * np.log(yPred), axis=1)
        average_loss = np.mean(loss)
        return average_loss

# Model oluştur
myNeuralNet = NeuralNetwork()
result = myNeuralNet.forward(np.random.rand(1, 784))
print(result)

# Diğer model
modelMNIST = NeuralNetwork(hidden_layers=[256])

# Dışarıdan yükleme - DİKKAT: Yolları raw string veya \\ ile yaz!
modelWeights = []
modelWeights.append(np.load(r"C:\Workspace\MNSIT\mymnıstc\modelWeights\layer_stack.1.weight.npy"))
modelWeights.append(np.load(r"C:\Workspace\MNSIT\mymnıstc\modelWeights\layer_stack.3.weight.npy"))

modelBiases = []
modelBiases.append(np.load(r"C:\Workspace\MNSIT\mymnıstc\modelWeights\layer_stack.1.bias.npy"))
modelBiases.append(np.load(r"C:\Workspace\MNSIT\mymnıstc\modelWeights\layer_stack.3.bias.npy"))

# Biçim kontrolü
for i in range(len(modelBiases)):
    print(f"Shapes PyTorchModel: {modelBiases[i].shape}")
    print(f"Shapes PyTorchModel Transformed: {np.expand_dims(modelBiases[i], axis=0).shape}")
    print(f"Shapes MyModel: {modelMNIST.biases[i].shape}")
    print()

for i in range(len(modelWeights)):
    print(f"Shapes PyTorchModel: {modelWeights[i].shape}")
    print(f"Shape PyTorchModel Transformed: {modelWeights[i].T.shape}")
    print(f"Shape MyModel: {modelMNIST.weights[i].shape}")
    print()