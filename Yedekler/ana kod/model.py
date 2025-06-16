import numpy as np
import os

base_dir = os.path.dirname(__file__)

class NeuralNetwork: 
    def __init__(self, input_size=784, hidden_layers=[512, 512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.gradientWeights = []
        self.gradientBiases = []
        self.iterations = 0

        # Katman başlatmaları
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))
        for i in range(len(hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i+1]))
            self.biases.append(np.zeros((1, hidden_layers[i+1])))
        self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))

    def forward(self, inputs):
        self.outputs = [inputs]
        for i in range(len(self.weights)):
            z = np.dot(self.outputs[-1], self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
                self.outputs.append(exp_scores / np.sum(exp_scores, axis=1, keepdims=True))
            else:
                self.outputs.append(np.maximum(0, z))
        return self.outputs[-1]

    def backwards(self, y_true):
        samples = len(self.outputs[-1])
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        d = self.outputs[-1].copy()
        d[range(samples), y_true] -= 1
        d /= samples
        self.gradientWeights = []
        self.gradientBiases = []

        d_input = d
        for i in reversed(range(len(self.weights))):
            prev_output = self.outputs[i]
            dW = np.dot(prev_output.T, d_input)
            dB = np.sum(d_input, axis=0, keepdims=True)
            self.gradientWeights.insert(0, dW)
            self.gradientBiases.insert(0, dB)

            if i > 0:
                d_input = np.dot(d_input, self.weights[i].T)
                d_input[self.outputs[i] <= 0] = 0

    def updateParams(self, lr=0.05, decay=1e-7):
        lr = lr * (1. / (1. + decay * self.iterations))
        for i in range(len(self.weights)):
            self.weights[i] -= lr * self.gradientWeights[i]
            self.biases[i] -= lr * self.gradientBiases[i]
        self.iterations += 1

    @staticmethod
    def LossCategoricalCrossEntropy(yPred, yTrue):
        yPred = np.clip(yPred, 1e-10, 1 - 1e-10)
        loss = -np.sum(yTrue * np.log(yPred), axis=1)
        return np.mean(loss)

    @staticmethod
    def sparse_to_one_hot(sparse_labels, num_classes):
        one_hot = np.zeros((len(sparse_labels), num_classes))
        one_hot[np.arange(len(sparse_labels)), sparse_labels] = 1
        return one_hot

def load_pretrained_model():
    model = NeuralNetwork(hidden_layers=[256, 128])
    weights = [np.load(os.path.join(base_dir, "modelWeights", f"final_weight_{i}.npy")) for i in range(3)]
    biases  = [np.load(os.path.join(base_dir, "modelWeights", f"final_bias_{i}.npy")) for i in range(3)]
    model.weights = weights
    model.biases = biases
    return model

def predict_image(model, image_array):
    image_array = image_array.reshape(1, 784)
    return model.forward(image_array).argmax()

def save_custom_example(image_array, label, save_dir="custom_data"):
    save_dir = os.path.join(base_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    index = 0
    while os.path.exists(os.path.join(save_dir, f"image_{index}.npy")):
        index += 1
    np.save(os.path.join(save_dir, f"image_{index}.npy"), image_array)
    np.save(os.path.join(save_dir, f"label_{index}.npy"), np.array([label]))
    print(f"[+] Kaydedildi: image_{index}.npy → label: {label}")

def load_custom_data(custom_dir="custom_data"):
    custom_dir = os.path.join(base_dir, custom_dir)
    images, labels = [], []
    index = 0
    while True:
        img_path = os.path.join(custom_dir, f"image_{index}.npy")
        lbl_path = os.path.join(custom_dir, f"label_{index}.npy")
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            break
        images.append(np.load(img_path))
        labels.append(np.load(lbl_path)[0])
        index += 1
    print(f"[+] Yüklenen custom veri: {len(images)} adet")
    return np.array(images), np.array(labels)