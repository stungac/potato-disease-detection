from tkinter import Tk, Button, Label, Frame, filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import json

def preprocess_image():
    """Preprocess the image for prediction"""
    global img_data
    img_array = np.array(img_data) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def display_prediction(predictions):
    """Display the predicted class and probability"""
    global result_label
    top_index = np.argmax(predictions)
    predicted_class = list(class_indices.keys())[top_index].replace('_', ' ')
    probability = predictions[0][top_index]
    result_label.config(text=f"Tahmin: {predicted_class}\nOlasılık: %{probability*100:.4f}")

def predict_image():
    """Predict the class of the image and display the result"""
    preprocessed_img = preprocess_image()
    predictions = model.predict(preprocessed_img)
    display_prediction(predictions)
    exeButton.config(text="Load the plant image", command=open_file)

def load_image(path):
    """Upload image for both model and Tkinter"""
    global img_data
    img_data = Image.open(path).convert('RGB').resize((224, 224))
    img_display = ImageTk.PhotoImage(img_data)
    return img_display

def open_file():
    """Ask the user to select an image and display it on the screen"""
    global img_label, img_display, result_label
    result_label.config(text="")
    file_path = filedialog.askopenfilename(
        title="Choose an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        img_display = load_image(file_path)
        img_label.config(image=img_display)
        img_label.image = img_display  # Tkinter'da garbage collection'a karşı koruma
        exeButton.config(text="Analize the Image", command=predict_image)

# Modelve sınıflar
model = tf.keras.models.load_model('../model/resnet_potato_disease_model.h5')
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Ana pencere
root = Tk()
root.title("Potato Disease Detection")
root.geometry("600x500")
root.configure(bg="#f7f7f7")

# Başlık
title_label = Label(root, text="Potato Disease Detector 🥔", font=("Helvetica", 18, "bold"), bg="#f7f7f7", fg="#2c3e50")
title_label.pack(pady=20)

# Görsel alanı
image_frame = Frame(root, bd=2, relief="solid", bg="white")
image_frame.pack(pady=10)
initial_path = "../data/pdd.png"
img_display = load_image(initial_path)
img_label = Label(image_frame, image=img_display, bg="white")
img_label.pack()

# Buton
exeButton = Button(
    root,
    text="Choose an Image",
    command=open_file,
    width=20,
    height=2,
    font=("Helvetica", 12, "bold"),
    bg="#27ae60",
    fg="green",
    activebackground="#2ecc71",
    cursor="hand2"
)
exeButton.pack(pady=20)

# Sonuç etiketi
result_label = Label(
    root,
    text="",
    font=("Helvetica", 14),
    bg="#f7f7f7",
    fg="#333333"
)
result_label.pack(pady=10)

root.mainloop()

