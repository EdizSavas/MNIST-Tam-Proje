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

        for i in range(len(self.weights)):
            assert self.weights[i].shape == self.gradientWeights[i].shape
            self.weights[i] -= lr * self.gradientWeights[i]

        for i in range(len(self.biases)):
            assert self.biases[i].shape == self.gradientBiases[i].shape
            self.biases[i] -= lr * self.gradientBiases[i]
                
        self.iterations += 1
        
    @staticmethod
    def LossCategoricalCrossEntropy(yPred, yTrue):
        yPred = np.clip(yPred, 1e-10, 1 - 1e-10)
        loss = -np.sum(yTrue * np.log(yPred), axis=1)
        average_loss = np.mean(loss)
        return average_loss
    
    @staticmethod
    def sparse_to_one_hot(sparse_labels, num_classes):
        one_hot_encoded = np.zeros((len(sparse_labels), num_classes))
        one_hot_encoded[np.arange(len(sparse_labels)), sparse_labels] = 1
        return one_hot_encoded
    
def load_pretrained_model():
    model = NeuralNetwork(hidden_layers=[256,128])
    weights = []
    biases = []
    for i in range(3):
        weights.append(np.load(os.path.join(base_dir, "modelWeights", f"final_weight_{i}.npy")))
        biases.append(np.load(os.path.join(base_dir, "modelWeights", f"final_bias_{i}.npy")))
    model.weights = weights
    model.biases = biases
    return model

def predict_image(model, image_array):
    image_array = image_array.reshape(1, 784)
    output = model.forward(image_array)
    return output.argmax()

def save_custom_example(image_array, label, save_dir="custom_data"):
    save_dir = os.path.join(base_dir, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    i = 0
    while True:
        img_path = os.path.join(save_dir, f"image_{i}.npy")
        lbl_path = os.path.join(save_dir, f"label_{i}.npy")
        if not os.path.exists(img_path) and not os.path.exists(lbl_path):
            break
        i += 1

    np.save(img_path, image_array)
    np.save(lbl_path, np.array([label]))
    print(f"[+] Kaydedildi: image_{i}.npy → label: {label}")

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


def load_custom_data(custom_dir="custom_data"):
    custom_dir = os.path.join(base_dir, custom_dir)
    images = []
    labels = []

    i = 0
    while True:
        img_path = os.path.join(custom_dir, f"image_{i}.npy")
        lbl_path = os.path.join(custom_dir, f"label_{i}.npy")
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            break
        img = np.load(img_path)
        lbl = np.load(lbl_path)[0]
        images.append(img)
        labels.append(lbl)
        i += 1

    print(f"[+] Yüklenen custom veri: {len(images)} adet")
    return np.array(images), np.array(labels)

def train_model():
    global myNeuralNet, accuracies, losses, save_dir, graf_path

    myNeuralNet = NeuralNetwork(hidden_layers=[256, 128])
    accuracies = []
    losses = []

    train_images = extract_images(os.path.join(base_dir, "MNISTdata", "train-images-idx3-ubyte.gz"))
    train_labels = extract_labels(os.path.join(base_dir, "MNISTdata", "train-labels-idx1-ubyte.gz"))
    test_images = extract_images(os.path.join(base_dir, "MNISTdata", "t10k-images-idx3-ubyte.gz"))
    test_labels = extract_labels(os.path.join(base_dir, "MNISTdata", "t10k-labels-idx1-ubyte.gz"))

    data = train_images.reshape(-1, 784).astype(np.float32)
    dataLabels = train_labels
    data = (data - 127.5) / 127.5

    custom_dir = os.path.join(base_dir, "custom_data")
    if os.path.exists(custom_dir):
        custom_imgs, custom_labels = load_custom_data(custom_dir)
        if len(custom_imgs) >= 300:
            print("[+] Custom veri dahil ediliyor.")
            custom_imgs = ((custom_imgs.astype(np.float32) - 127.5) / 127.5).reshape(len(custom_imgs), 784)
            data = np.concatenate([data, custom_imgs])
            dataLabels = np.concatenate([dataLabels, custom_labels])
        else:
            print(f"[!] Custom veri ({len(custom_imgs)}) yetersiz. Eğitim dışı bırakıldı.")

    BATCH_SIZE = 32
    best_loss = float("inf")
    patience = 20
    no_improve = 0
    delta = 1e-6
    print("Toplam eğitim verisi:", data.shape[0])

    for epoch in range(1, 10):
        train_steps = len(data) // BATCH_SIZE
        for step in range(train_steps):
            batch_X = data[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            batch_Y = dataLabels[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            X, Y = batch_X, batch_Y

            output = myNeuralNet.forward(X)
            predicitons = np.argmax(output, axis=1)
            if len(Y.shape) == 2:
                Y = np.argmax(Y, axis=1)

            accuracy = np.mean(predicitons == Y)
            loss = myNeuralNet.LossCategoricalCrossEntropy(output, myNeuralNet.sparse_to_one_hot(Y, 10))

            if step % 100 == 0:
                accuracies.append(accuracy)
                losses.append(loss)
                print(f"epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}")
                if best_loss - loss > delta:
                    best_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print("Erken durdurma tetiklendi.")
                    break

            myNeuralNet.backwards(Y)
            myNeuralNet.updateParams(lr=0.05, decay=1e-6)

    # Model kaydet
    save_dir = os.path.join(base_dir, "modelWeights")
    os.makedirs(save_dir, exist_ok=True)
    for i, weight in enumerate(myNeuralNet.weights):
        np.save(os.path.join(save_dir, f"final_weight_{i}.npy"), weight)
    for i, bias in enumerate(myNeuralNet.biases):
        np.save(os.path.join(save_dir, f"final_bias_{i}.npy"), bias)

    # Grafik çiz
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(accuracies, 'b-', label='Doğruluk')
    ax2.plot(losses, 'orange', label='Kayıp')
    ax1.set_xlabel("Kayıt Noktaları (her 100 adımda bir)")
    ax1.set_ylabel("Doğruluk", color='b')
    ax2.set_ylabel("Kayıp", color='orange')
    plt.title("Eğitim Doğruluğu ve Kayıp Değerleri")
    ax1.grid(True)
    fig.tight_layout()
    graf_dir = os.path.join(base_dir, "graph")
    os.makedirs(graf_dir, exist_ok=True)
    existing = [int(f.split('.')[0]) for f in os.listdir(graf_dir) if f.endswith('.png') and f.split('.')[0].isdigit()]
    next_index = max(existing) + 1 if existing else 0
    graf_path = os.path.join(graf_dir, f"{next_index}.png")
    fig.savefig(graf_path)
    print(f"[+] Grafik kaydedildi: {graf_path}")

    return accuracies[-1], losses[-1], graf_path, save_dir


if __name__ == "__main__":
    max_trials = 5
    min_accuracy = 0.98
    max_loss = 0.1

    for trial in range(max_trials):
        print(f"\n==== Deneme #{trial+1} Başlıyor ====")
        acc, loss, graf_path, save_dir = train_model()

        if acc >= min_accuracy and loss <= max_loss:
            print(f"[✓] Başarılı model bulundu. Kayıt ediliyor.")
            save_path = os.path.join(base_dir, "best_model")
            os.makedirs(save_path, exist_ok=True)
            for i, weight in enumerate(myNeuralNet.weights):
                np.save(os.path.join(save_path, f"final_weight_{i}.npy"), weight)
            for i, bias in enumerate(myNeuralNet.biases):
                np.save(os.path.join(save_path, f"final_bias_{i}.npy"), bias)
            break
        else:
            print(f"[✗] Kötü sonuç. Dosyalar temizleniyor...")
            try:
                os.remove(graf_path)
            except Exception:
                pass
            for f in os.listdir(save_dir):
                try:
                    os.remove(os.path.join(save_dir, f))
                except Exception:
                    pass
                
