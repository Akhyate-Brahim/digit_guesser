import tkinter as tk
import tensorflow as tf
import os
import joblib
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw
import numpy as np
from sklearn import linear_model, tree
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.datasets import mnist

import logging
logging.basicConfig(level=logging.INFO)

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)) / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)) / 255.0

# Create and train the models
models = {
    "Logistic Regression": linear_model.LogisticRegression(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "ANN": MLPClassifier(hidden_layer_sizes=(128,)),
    "CNN": Sequential([
        Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size=3, strides=1, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Dropout(0.25),
        Conv2D(filters=64, kernel_size=3, strides=1, activation='relu', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Dropout(0.25),
        Flatten(),
        Dense(units=128, use_bias=False),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.25),
        Dense(units=10, activation='softmax')
    ])
}

for name, model in models.items():
    model_path = f"./models/{name.lower()}_model.h5"
    if os.path.exists(model_path):
        logging.info(f"Loading pre-trained {name} model from disk...")
        if name == "CNN":
            model = tf.keras.models.load_model(model_path)  # Load the entire model
            _, accuracy = model.evaluate(x_test, y_test)  # Use evaluate() for CNN model
        else:
            model = joblib.load(model_path)
            accuracy = model.score(x_test.reshape((10000, 28 * 28)), y_test)
    else:
        logging.info(f"Training {name} model...")
        if name == "CNN":
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
            model.save(model_path)  # Save the entire model
            _, accuracy = model.evaluate(x_test, y_test)
        else:
            model.fit(x_train.reshape((60000, 28 * 28)), y_train)
            joblib.dump(model, model_path)
            accuracy = model.score(x_test.reshape((10000, 28 * 28)), y_test)
    logging.info(f"{name} accuracy: {accuracy:.4f}")

class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("Number Guesser")
        
        # Create the main frame
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the drawing canvas
        self.canvas = tk.Canvas(self.main_frame, width=280, height=280, bg="white")
        self.canvas.pack(side=tk.TOP, padx=10, pady=10)
        
        # Create the buttons frame
        self.buttons_frame = tk.Frame(self.main_frame)
        self.buttons_frame.pack(side=tk.TOP, padx=10, pady=10)
        
        # Create the model selection dropdown
        self.model_var = tk.StringVar(value="CNN")
        self.model_dropdown = ttk.Combobox(self.buttons_frame, textvariable=self.model_var, values=list(models.keys()), state="readonly")
        self.model_dropdown.pack(side=tk.LEFT, padx=5)
        
        # Create the guess button
        self.guess_button = tk.Button(self.buttons_frame, text="Guess", command=self.guess_number)
        self.guess_button.pack(side=tk.LEFT, padx=5)
        
        # Create the clear button
        self.clear_button = tk.Button(self.buttons_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Create the image object for drawing
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind the mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.draw_lines)
    
    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill="black")
    
    def guess_number(self):
        img = self.image.resize((28, 28)).convert("L")
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)
        
        model = models[self.model_var.get()]
        prediction = model.predict(img)
        guess = np.argmax(prediction)
        messagebox.showinfo("Guess", f"I think you drew a {guess}!")
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

root = tk.Tk()
app = DrawingApp(root)
root.mainloop()