#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def load_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        print(f"Video loaded: {file_path}")

def start_classification():
    print("Classification started")
    messagebox.showinfo("Result", "Classification Complete!")

def exit_application():
    root.destroy()

# Set up the main application window
root = tk.Tk()
root.title("Yoga Pose Classification")

# Create and place buttons
load_button = tk.Button(root, text="Load Video", command=load_video)
load_button.pack()

start_button = tk.Button(root, text="Start Classification", command=start_classification)
start_button.pack()

exit_button = tk.Button(root, text="Exit", command=exit_application)
exit_button.pack()

# Start the GUI event loop
root.mainloop()

