import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv(r"C:\Users\Asus\Downloads\ML DATSETS\BAnk\Churn_Modelling.csv")

# Inspect the data
print(df.head())
print(df.info())




# 1. Define Target (y) and Features (X)
y = df['Exited']
# Drop irrelevant columns and the target variable
X = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)

# 2. Encode Categorical Features
# 'Geography' (3 values) and 'Gender' (2 values) must be converted to numbers.
# We'll use one-hot encoding for this.
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

# After encoding, X will have new columns like 'Geography_Germany', 
# 'Geography_Spain', and 'Gender_Male' (all 0s or 1s).

# 3. Split the data into training and test sets
# We use an 80/20 split. 
# 'stratify=y' ensures the proportion of churners (Exited=1) is the
# same in both the train and test sets, which is crucial for imbalanced datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")







# Identify columns that need scaling (all except the one-hot encoded dummies)
# Get a list of the dummy columns
dummy_cols = [col for col in X_train.columns if 'Geography_' in col or 'Gender_' in col]
# Get a list of columns to scale
cols_to_scale = [col for col in X_train.columns if col not in dummy_cols]

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])

# Only transform the test data (using the scaler fit on X_train)
X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

# Display the scaled data
print(X_train.head())







# Get the number of input features
input_dim = X_train.shape[1] 

# Build the Sequential model
model = keras.Sequential([
    # Input layer
    keras.layers.Input(shape=(input_dim,)),
    
    # First hidden layer
    keras.layers.Dense(12, activation='relu'),
    
    # Second hidden layer
    keras.layers.Dense(8, activation='relu'),
    
    # Output layer
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',                # 'adam' is a good default optimizer
    loss='binary_crossentropy',      # Standard loss function for binary classification
    metrics=['accuracy']             # Metric to monitor
)

# Display the model's architecture
model.summary()










# Train the model
# We use 'validation_split' to automatically set aside a portion of the
# training data (e.g., 10%) to monitor validation loss and check for overfitting.
history = model.fit(
    X_train, 
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model on the test set
print("\n--- Model Evaluation ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Get detailed predictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to 0 or 1

# Print a detailed classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Did not Leave (0)', 'Left (1)']))
