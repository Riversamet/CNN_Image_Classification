#!/usr/bin/env python3

"""
Predict Chinese traffic sign type for one or multiple Chinese traffic signs 
at once using the CNN model with the best metrics from 
main.ipynb. Display predicted type and confidence.

Inputs:
    Path to saved CNN keras model
    Paths to all traffic sign images desired to be categorized

Outputs:
    matplotlib.pyplot pop-up with categorization results for each Chinese
    traffic sign image input.
"""

from tensorflow.keras.models import load_model
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from labels import LABELS as labels

model_path = sys.argv[1]
traffic_sign_paths = sys.argv[2:]

model = load_model(model_path)

results = []

for sign in traffic_sign_paths:
    rgb_img = Image.open(sign).convert('RGB')
    resize_img = rgb_img.resize((32, 32))
    resize_img_final = np.array(resize_img.resize((96, 96)))    # Resize to 32x32 
                                                                # then 96x96 as 
                                                                # was done for 
                                                                # MNV2 model 
                                                                # training
    final_img = resize_img_final / 255.0

    # Expand dims for batch process (first dim is batch size)
    final_img_ar = np.expand_dims(final_img, axis=0)

    pred = model.predict(final_img_ar)[0]

    pred_choice = np.argmax(pred)
    confidence = float(pred[pred_choice])
    label = labels[pred_choice]
    results.append([final_img, label, confidence, sign])

idx = [0]

# Print each sign

fig, ax = plt.subplots(figsize=(5, 5.5))

def draw():
    img, label, conf, path = results[idx[0]]
    ax.clear()
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"[{idx[0]+1}/{len(results)}]\n{label}\n({conf:.1%})\nUse ← → arrow keys", fontsize=11)
    fig.canvas.draw_idle()

# Allow arrow keys to switch between traffic signs in pop-up window

def on_key(event):
    if event.key == 'right':
        idx[0] = (idx[0] + 1) % len(results)
        draw()
    elif event.key == 'left':
        idx[0] = (idx[0] - 1) % len(results)
        draw()

fig.canvas.mpl_connect('key_press_event', on_key)
draw()
plt.show()