#importing Necessary Libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np

# Load the model
from tensorflow.keras.models import load_model  # type: ignore
model = load_model('Age_Sex_Detection.keras')

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Age & Gender Detector')
top.configure(background='#CDCDCD')

# Initialize the labels (1 for age and 1 for Sex)
label1 = Label(top, background="#cdcdcd", font=('arial', 15, "bold"))
label2 = Label(top, background="#cdcdcd", font=('arial', 15, "bold"))
sign_image = Label(top)

def Detect(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((48, 48))  # Correctly resize the image
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    print(f"Preprocessed image shape: {image.shape}")

    # Ensure image has the correct dimensions
    if image.shape[-1] == 4:  # If image has an alpha channel, remove it
        image = image[:, :, :, :3]
    image = image / 255.0  # Normalize image

    # Debugging: print preprocessed image stats
    print(f"Image shape after preprocessing: {image.shape}")
    print(f"Image pixel values range: {image.min()} to {image.max()}")

    # Predict age and gender
    pred = model.predict(image)
    print(f"Raw model predictions: {pred}")

    age = int(np.round(pred[1][0]))  # Assuming age prediction is the second output
    sex_prob = pred[0][0]  # Assuming gender prediction is the first output

    # Interpret gender prediction
    sex = int(np.round(sex_prob))
    sex_f = ["Male", "Female"]

    print("Predicted age is " + str(age))
    print("Predicted Gender is " + sex_f[sex])
    label1.configure(foreground="#011638", text=f"Predicted age: {age}")
    label2.configure(foreground="#011638", text=f"Predicted Gender: {sex_f[sex]}")




# Define Show_detect button function
def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

# Define Upload Image Function
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        label2.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error: {e}")

upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)
label2.pack(side='bottom', expand=True)
heading = Label(top, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#cdcdcd", foreground="#364156")
heading.pack()
top.mainloop()



    
