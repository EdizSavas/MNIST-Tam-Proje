import numpy as np 
import matplotlib.pyplot as plt
import gzip 
import os 

base_dir = os.path.dirname(__file__)
print(base_dir)

class NeuralNetwork: #Sınıf tanımı
    def __init__(self, input_size = 784, hidden_layers = [512,512], output_size = 10): #Sınıftan nesne oluşunca otomatik çalışır

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
            
        self.weights.append(0.01 * np.random.randn(hidden_layers[len(hidden_layers) - 1], output_size))
        self.biases.append(np.zeros((1, output_size)))
            
    def forward(self, inputs): #Giriş verisi alır ve katmanlar boyunca ilerleyerek son çıktıyı üretir
        layers = [inputs]
        for i in range(len(self.weights)):
            layers.append(np.dot(layers[-1], self.weights[i]) + self.biases[i])
            if i == len(self.weights) - 1:
                finalOutput = np.exp(layers[-1] - np.max(layers[-1], axis=1, keepdims=True))
                finalOutput = finalOutput / np.sum(finalOutput, axis=1, keepdims= True)
                layers.append(finalOutput)
            else:
                layers.append(np.maximum(0, layers[-1]))
        return layers[-1]
    
    @staticmethod
    def LossCategoricalCrossEntropy(yPred, yTrue):
        yPred = np.clip(yPred, 1e-10, 1 - 1e-10)
        loss = -np.sum(True * np.log(yPred), axis=1)
        average_loss = np.mean(loss)
        return average_loss
    
myNeuralNet =  NeuralNetwork()
result = myNeuralNet.forward(np.random.rand(1, 784))
print(result)

modelMNIST = NeuralNetwork(hidden_layers=[256])

modelWeights = []
modelWeights.append(np.load(os.path.join(base_dir, "modelWeights", "layer_stack.1.weight.npy")))
modelWeights.append(np.load(os.path.join(base_dir, "modelWeights", "layer_stack.3.weight.npy")))

modelBiases = []
modelBiases.append(np.load(os.path.join(base_dir, "modelWeights", "layer_stack.1.bias.npy")))
modelBiases.append(np.load(os.path.join(base_dir, "modelWeights", "layer_stack.3.bias.npy")))

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

for i in range(len(modelBiases)):
    modelMNIST.biases[i] = np.expand_dims(modelBiases[i], axis=0)

for i in range(len(modelWeights)):
    modelMNIST.weights[i] = modelWeights[i].T

def extract_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=np.uint32, count=4).byteswap()
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def extract_labels(filename):
    with gzip.open(filename, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype=np.uint32, count=2).byteswap()
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

train_images = extract_images(os.path.join(base_dir, "MNISTdata", "train-images-idx3-ubyte.gz"))
train_labels = extract_labels(os.path.join(base_dir, "MNISTdata", "train-labels-idx1-ubyte.gz"))
test_images = extract_images(os.path.join(base_dir, "MNISTdata", "t10k-images-idx3-ubyte.gz"))
test_labels = extract_labels(os.path.join(base_dir, "MNISTdata", "t10k-labels-idx1-ubyte.gz"))

testImage1 =np.expand_dims(test_images[0].flatten(), axis=0)
print(f"Test Label: {test_labels[0]}")
print(f"Predict Label: {modelMNIST.forward(testImage1)}")

correct = 0 
incorrect = 0 
for i in range(len(test_images)):
    image = np.expand_dims(test_images[i].flatten(), axis=0)

    if int(test_labels[i]) == int(modelMNIST.forward(image).argmax()):
        correct += 1 
    else:
        incorrect += 1

print((correct)/(correct+incorrect))
