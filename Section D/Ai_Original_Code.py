import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# --------------------------------------------------
# 1. Create synthetic food delivery dataset
# --------------------------------------------------

np.random.seed(42)

samples = 600

hour_of_day = np.random.randint(0, 24, samples)
distance_km = np.random.uniform(1, 30, samples)
order_value = np.random.uniform(100, 2000, samples)


# --------------------------------------------------
# 2. Create three priority classes
# --------------------------------------------------

priority = np.zeros(samples, dtype=int)

for i in range(samples):

    score = (
        hour_of_day[i]
        + distance_km[i]
        + order_value[i] / 200
    )

    if score < 20:
        priority[i] = 0       # Low

    elif score < 35:
        priority[i] = 1       # Medium

    else:
        priority[i] = 2       # High


# One-hot encode target
y = to_categorical(priority, num_classes=3)

# Combine input features
X = np.column_stack([
    hour_of_day,
    distance_km,
    order_value
])


# --------------------------------------------------
# 3. Build the neural network
# --------------------------------------------------

model = Sequential([
    Dense(64, activation="relu", input_shape=(3,)),
    Dense(32, activation="relu"),
    Dense(3, activation="sigmoid")
])


# --------------------------------------------------
# 4. Compile the model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

history = model.fit(
    X,
    y,
    epochs=20,
    validation_split=0.2,
    verbose=1
)


# --------------------------------------------------
# 6. Print training accuracy after every epoch
# --------------------------------------------------

for epoch, accuracy in enumerate(
    history.history["accuracy"],
    start=1
):

    print(
        f"Epoch {epoch}: "
        f"Training Accuracy = {accuracy:.4f}"
    )


# --------------------------------------------------
# 7. Plot training and validation loss
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()


plt.subplot(1, 2, 2)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()


# --------------------------------------------------
# 8. Save the trained model
# --------------------------------------------------

model.save("food_priority_model.keras")

print(
    "Model saved as food_priority_model.keras"
)


# --------------------------------------------------
# 9. Reload the saved model
# --------------------------------------------------

loaded_model = load_model(
    "food_priority_model.keras"
)

print("Model successfully reloaded.")


# --------------------------------------------------
# 10. Make one test prediction
# --------------------------------------------------

test_input = np.array([
    [18, 12.5, 850]
])

prediction = loaded_model.predict(
    test_input,
    verbose=0
)

predicted_class = np.argmax(
    prediction[0]
)

classes = [
    "Low",
    "Medium",
    "High"
]

print(
    "Test Prediction:",
    classes[predicted_class]
)