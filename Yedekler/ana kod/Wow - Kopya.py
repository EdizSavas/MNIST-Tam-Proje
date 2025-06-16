import numpy as np 
import matplotlib.pyplot as plt
import gzip 
import os 

base_dir = os.path.dirname(__file__)
#print(base_dir)

class NeuralNetwork: 
    def __init__(self, input_size = 784, hidden_layers = [512,512], output_size = 10):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.gradientWeights = []
        self.gradientBiases = []
        self.iterations = 0

        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))
        
        for i in range(len(hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i + 1]))
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))
            
        self.weights.append(0.01 * np.random.randn(hidden_layers[len(hidden_layers) - 1], output_size))
        self.biases.append(np.zeros((1, output_size)))
                    
    def forward(self, inputs): 
        self.outputs = [inputs]
        self.outputsTesting = ["inputs"]
        
        for i in range(len(self.weights)):
            self.outputs.append(np.dot(self.outputs[-1], self.weights[i]) + self.biases[i])
            self.outputsTesting.append("dense")
            
            if i == len(self.weights) - 1:
                finalOutput = np.exp(self.outputs[-1] - np.max(self.outputs[-1], axis=1, keepdims=True))
                finalOutput = finalOutput / np.sum(finalOutput, axis=1, keepdims= True)
                self.outputs.append(finalOutput)
                self.outputsTesting.append("softmax")
            else:
                self.outputs.append(np.maximum(0, self.outputs[-1]))
                self.outputsTesting.append("relu")
                
        return self.outputs[-1]
    
    def backwards(self, y_true):
        samples = len(self.outputs[-1])
        
        if len(y_true.shape) == 2:
            #print("Chaning to Discrete Values")
            y_true = np.argmax(y_true, axis=1)
            
        dSoftMaxCrossEntropy = self.outputs[-1].copy()
        dSoftMaxCrossEntropy[range(samples), y_true] -= 1
        dSoftMaxCrossEntropy = dSoftMaxCrossEntropy / samples
        
        dInputs = np.dot(dSoftMaxCrossEntropy.copy(), self.weights[-1].T)
        
        dWeights = np.dot(self.outputs[-3].T, dSoftMaxCrossEntropy.copy())
        dBiases = np.sum(dSoftMaxCrossEntropy.copy(), axis=0, keepdims=True)
        self.gradientWeights = [dWeights] + self.gradientWeights
        self.gradientBiases = [dBiases] + self.gradientBiases
        
        i = -3
        j = -1 
        for _ in range(len(self.hidden_layers)):
            i -= 1
            j -= 1 
            
            dInputsReLU = dInputs.copy()
            dInputsReLU[self.outputs[i] <= 0] = 0
            
            i -= 1
            dInputs = np.dot(dInputsReLU, self.weights[j].T)
            dWeights = np.dot(self.outputs[i].T, dInputsReLU)
            dBiases = np.sum(dInputsReLU, axis=0, keepdims=True)
            self.gradientWeights = [dWeights] + self.gradientWeights
            self.gradientBiases = [dBiases] + self.gradientBiases
                    
            #print("dense1.dweights")
            #print(dWeights)
            #print(dWeights.shape)
            #print("dense1.dbiases")
            #print(dBiases)
            #print(dBiases.shape)
            
    def updateParams(self, lr=0.05, decay= 1e-7):
        lr = lr * (1. / (1. + decay * self.iterations))

        for i in range(len(self.weights)-1):
            if i != len(self.weights)-1:
                assert self.weights[i].shape == self.gradientWeights[i].shape
                self.weights[i] += -lr*self.gradientWeights[i]
        
        for i in range(len(self.biases)-1):
            if i != len(self.biases[i])-1:
                assert self.biases[i].shape == self.gradientBiases[i].shape
                self.biases[i] += -lr*self.gradientBiases[i]
                
        self.iterations += 1
        
    @staticmethod
    def LossCategoricalCrossEntropy(yPred, yTrue):
        yPred = np.clip(yPred, 1e-10, 1 - 1e-10)
        loss = -np.sum(True * np.log(yPred), axis=1)
        average_loss = np.mean(loss)
        return average_loss
    
    @staticmethod
    def sparse_to_one_hot(sparse_labels, num_classes):
        one_hot_encoded = np.zeros((len(sparse_labels), num_classes))
        one_hot_encoded[np.arange(len(sparse_labels)), sparse_labels] = 1
        return one_hot_encoded
    
def load_pretrained_model():
    
    model = NeuralNetwork(hidden_layers=[256])
    base_dir = os.path.dirname(__file__)
    
    weights = [
        np.load(os.path.join(base_dir, "modelWeights", "layer_stack.1.weight.npy")),
        np.load(os.path.join(base_dir, "modelWeights", "layer_stack.3.weight.npy"))
    ]
    biases = [
        np.load(os.path.join(base_dir, "modelWeights", "layer_stack.1.bias.npy")),
        np.load(os.path.join(base_dir, "modelWeights", "layer_stack.3.bias.npy"))
    ]

    for i in range(len(weights)):
        model.weights[i] = weights[i].T
        model.biases[i] = np.expand_dims(biases[i], axis=0)
        
    return model

def predict_image(model, image_array):
    image_array = image_array.reshape(1, 784)
    output = model.forward(image_array)
    return output.argmax()

if __name__ == "__main__":
    myNeuralNet =  NeuralNetwork(hidden_layers=[256,128])

    result = myNeuralNet.forward(np.random.rand(1, 784))
    #print(result)

    modelMNIST = NeuralNetwork(hidden_layers=[256])

    modelWeights = []
    modelWeights.append(np.load(os.path.join(base_dir, "modelWeights", "layer_stack.1.weight.npy")))
    modelWeights.append(np.load(os.path.join(base_dir, "modelWeights", "layer_stack.3.weight.npy")))

    modelBiases = []
    modelBiases.append(np.load(os.path.join(base_dir, "modelWeights", "layer_stack.1.bias.npy")))
    modelBiases.append(np.load(os.path.join(base_dir, "modelWeights", "layer_stack.3.bias.npy")))

    #for i in range(len(modelBiases)):
        #print(f"Shapes PyTorchModel: {modelBiases[i].shape}")
        #print(f"Shapes PyTorchModel Transformed: {np.expand_dims(modelBiases[i], axis=0).shape}")
        #print(f"Shapes MyModel: {modelMNIST.biases[i].shape}")
        #print()

    #for i in range(len(modelWeights)):
        #print(f"Shapes PyTorchModel: {modelWeights[i].shape}")
        #print(f"Shape PyTorchModel Transformed: {modelWeights[i].T.shape}")
        #print(f"Shape MyModel: {modelMNIST.weights[i].shape}")
        #print()

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
    #print(f"Test Label: {test_labels[0]}")
    #print(f"Predict Label: {modelMNIST.forward(testImage1)}")

    correct = 0 
    incorrect = 0 
    for i in range(len(test_images)):
        image = np.expand_dims(test_images[i].flatten(), axis=0)

        if int(test_labels[i]) == int(modelMNIST.forward(image).argmax()):
            correct += 1 
        else:
            incorrect += 1

    #print((correct)/(correct+incorrect))

    best_loss = float("inf")
    patience = 5
    no_improve = 0
    
    data = train_images
    dataLabels = train_labels

    data = (data.astype(np.float32)-127.5)/127.5

    data = data.reshape(60000, 784)

    accuracies = []
    losses = []

    BATCH_SIZE = 32

    for epoch in range(1, 10):
        #print(f"epoch: {epoch}")
        train_steps = len(data) // BATCH_SIZE
    
        for step in range(train_steps):
            batch_X = data[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            batch_Y = dataLabels[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        
            X = batch_X
            Y = batch_Y
        
            output = myNeuralNet.forward(X)
        
            if step % 100 == 0: 
                predicitons = np.argmax(output, axis=1)
                if len(Y.shape) == 2:
                    Y = np.argmax(Y, axis=1)
                accuracy = np.mean(predicitons==Y)
            
                loss = myNeuralNet.LossCategoricalCrossEntropy(output, myNeuralNet.sparse_to_one_hot(Y,10))
                accuracies.append(accuracy)
                losses.append(loss)
            
                print(f"epoch: {epoch}, " + 
                    f"acc: {accuracy:.3f}, " + 
                    f"loss: {loss:.3f}")
                
                if loss < best_loss:
                    best_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print("Erken durdurma tetiklendi.")
                break
        
            myNeuralNet.backwards(Y)
        
            myNeuralNet.updateParams(lr=0.05, decay=1e-6)
            #myNeuralNet.updateParams(lr=0.5, decay=1e-6)
            
            np.save("modelWeights/final_weights.npy", myNeuralNet.weights)
            np.save("modelWeights/final_biases.npy", myNeuralNet.biases)
        
        
    dataTest = test_images
    dataTestLabels = test_labels

    dataTest = (dataTest.astype(np.float32)-127.5)/127.5
    #print(dataTest.shape)
    dataTest = dataTest.reshape(10000, 784)

    X = dataTest
    Y = dataTestLabels

    output = myNeuralNet.forward(dataTest)

    predicitons = np.argmax(output, axis=1)
    if len(Y.shape) == 2:
        Y = np.argmax(y, axis=1)
    accuracy = np.mean(predicitons==Y)
    #print(accuracy)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()

    ax1.plot(accuracies, 'b-', label='Doğruluk')
    ax2.plot(losses, 'orange', label='Kayıp')

    ax1.set_xlabel("Kayıt Noktaları (her 100 adımda bir)")
    ax1.set_ylabel("Doğruluk (%)", color='b')
    ax2.set_ylabel("Kayıp", color='orange')

    plt.title("Eğitim Doğruluğu ve Kayıp Değerleri")
    ax1.grid(True)
    fig.tight_layout()
    plt.show()


