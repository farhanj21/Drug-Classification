import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Loss function
def categorical_crossentropy(predictions, targets):
    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    N = predictions.shape[0]
    log_likelihood = -np.log(predictions[range(N), targets])
    loss = np.sum(log_likelihood) / N
    return loss

# Gradient of loss function
def softmax_cross_entropy_loss_gradient(predictions, targets):
    N = predictions.shape[0]
    predictions[range(N), targets] -= 1
    return predictions / N

# Neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros((1, output_size))

    def forward(self, X):
        self.z1 = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.bias_hidden_output
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        delta_z2 = self.a2 - y
        delta_weights_hidden_output = np.dot(self.a1.T, delta_z2)
        delta_bias_hidden_output = np.sum(delta_z2, axis=0, keepdims=True)

        delta_z1 = np.dot(delta_z2, self.weights_hidden_output.T) * (self.a1 * (1 - self.a1))
        delta_weights_input_hidden = np.dot(X.T, delta_z1)
        delta_bias_input_hidden = np.sum(delta_z1, axis=0, keepdims=True)

        self.weights_input_hidden -= learning_rate * delta_weights_input_hidden
        self.bias_input_hidden -= learning_rate * delta_bias_input_hidden
        self.weights_hidden_output -= learning_rate * delta_weights_hidden_output
        self.bias_hidden_output -= learning_rate * delta_bias_hidden_output

    def train(self, X, y, epochs, learning_rate):
        y_onehot = np.eye(self.weights_hidden_output.shape[1])[y]
        for epoch in range(epochs):
            output = self.forward(X)
            loss = categorical_crossentropy(output, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
            self.backward(X, y_onehot, learning_rate)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Load and preprocess the dataset
data = pd.read_csv("./archive/drug200.csv")

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data['Age'] = imputer.fit_transform(data[['Age']])

# Encode categorical variables
label_encoder_sex = LabelEncoder()  
label_encoder_bp = LabelEncoder()  
label_encoder_chol = LabelEncoder()  
data['Sex'] = label_encoder_sex.fit_transform(data['Sex'])
data['BP'] = label_encoder_bp.fit_transform(data['BP'])
data['Cholesterol'] = label_encoder_chol.fit_transform(data['Cholesterol'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['Age', 'Na_to_K']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Split the dataset into training and testing sets
X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = data['Drug'] 

# Initialize the label encoder for 'Drug'
label_encoder_drug = LabelEncoder()
y_encoded = label_encoder_drug.fit_transform(y) 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the neural network
input_size = X_train.shape[1]
hidden_size = 8  
output_size = len(label_encoder_drug.classes_)  
learning_rate = 0.1
epochs = 1000

model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X_train.to_numpy(), y_train, epochs, learning_rate)

# Evaluate the model
predictions_train = model.predict(X_train.to_numpy())
accuracy_train = np.mean(predictions_train == y_train)
print("Train Accuracy:", accuracy_train)

predictions_test = model.predict(X_test.to_numpy())
accuracy_test = np.mean(predictions_test == y_test)
print("Test Accuracy:", accuracy_test)

# Prediction and Drug Prescription for a new patient
new_patient_data = pd.DataFrame({'Age': [40], 'Sex': ['M'], 'BP': ['HIGH'], 'Cholesterol': ['HIGH'], 'Na_to_K': [10]})
new_patient_data[numerical_features] = scaler.transform(new_patient_data[numerical_features])

# Use the same LabelEncoder instance used for 'Sex', 'BP', and 'Cholesterol' columns during training
new_patient_data['Sex'] = label_encoder_sex.transform(new_patient_data['Sex'])
new_patient_data['BP'] = label_encoder_bp.transform(new_patient_data['BP'])
new_patient_data['Cholesterol'] = label_encoder_chol.transform(new_patient_data['Cholesterol'])

predicted_drug_index = model.predict(new_patient_data.to_numpy())
predicted_drug = label_encoder_drug.inverse_transform(predicted_drug_index)[0]
print("Predicted Drug for the new patient:", predicted_drug)
