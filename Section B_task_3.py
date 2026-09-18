import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# 1. Generate synthetic dataset
# ---------------------------------------------------------

np.random.seed(42)

n_samples = 600

hour_of_day = np.random.randint(0, 24, n_samples)
day_of_week = np.random.randint(0, 7, n_samples)
temperature_celsius = np.random.uniform(10, 45, n_samples)

# Create a demand score using the three features
demand_score = (
    2.0 * np.sin((hour_of_day - 8) * np.pi / 12)
    + 0.5 * (day_of_week >= 5)
    + 0.03 * (temperature_celsius - 25)
    + np.random.normal(0, 0.3, n_samples)
)

# Convert score into three classes:
# 0 = Low, 1 = Medium, 2 = High
labels = np.digitize(
    demand_score,
    bins=[-0.2, 0.5]
)

class_names = ["Low", "Medium", "High"]

# Feature matrix
X = np.column_stack([
    hour_of_day,
    day_of_week,
    temperature_celsius
])

# One-hot encode target
y = to_categorical(labels, num_classes=3)


# ---------------------------------------------------------
# 2. Split dataset into training and test sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


# ---------------------------------------------------------
# 3. Scale input features
# ---------------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------------
# 4. Build Keras Sequential ANN
# ---------------------------------------------------------

model = Sequential([
    Dense(64, activation="relu", input_shape=(3,)),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.3),

    Dense(3, activation="softmax")
])


# ---------------------------------------------------------
# 5. Compile the model
# ---------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ---------------------------------------------------------
# 6. Train the model
# ---------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_split=0.2,
    batch_size=32,
    verbose=1
)


# ---------------------------------------------------------
# 7. Evaluate the model on the held-out test set
# ---------------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nFinal Test Accuracy:")
print(f"{test_accuracy * 100:.2f}%")


# ---------------------------------------------------------
# 8. Plot training vs validation accuracy
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.show()
