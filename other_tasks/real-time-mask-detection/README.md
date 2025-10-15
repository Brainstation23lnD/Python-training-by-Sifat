### **Real-time Mask Detection**

**Description:**
This project implements a **real-time face mask detection system** using **Convolutional Neural Networks (CNNs)**. It trains on the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) to classify faces as either **Mask** or **No Mask**.

After training, the model can be deployed to a real-time application using **OpenCV** or integrated into a **FastAPI backend** for web or API-based usage.
---

### **Topics Covered**

1. **Convolutional Neural Networks (CNNs)**
   * CNNs are neural networks specifically designed for **image data**.
   * They automatically extract features (edges, shapes, patterns) from images using **convolutional layers**, **pooling layers**, and **fully connected layers**.
   * In this project, a simple CNN is used first as a baseline for mask detection.

2. **Transfer Learning with VGG16 & ResNet**
   * **VGG16**: A deep CNN with 16 layers, widely used for image classification. Transfer learning allows using pretrained weights from ImageNet to achieve faster and more accurate training on smaller datasets.
   * **ResNet**: Uses **skip connections** to allow training of very deep networks without vanishing gradients. Offers high accuracy and robustness for image classification tasks.

3. **Real-time Detection**
   * After training, the model can process live webcam feeds to detect masked and unmasked faces.
   * Uses **OpenCV** for video capture and face detection.

4. **FastAPI Deployment (planned)**
   * Model can later be served via **FastAPI** for web or API integration.
   * The API will accept images or video frames and return mask detection predictions in real time.

---
### **Dataset**
* **Face Mask Detection Dataset (Kaggle):**
  * Images of faces with and without masks.
  * Supports both training and validation splits.
  * [Dataset Link](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
---
### **Installation / Requirements**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
# Install dependencies
pip install -r requirements.txt
```
**requirements.txt sample:**
```
tensorflow==2.14.0
keras==2.14.0
opencv-python
numpy
matplotlib
scikit-learn
```
---

### **Usage**
#### **1. Train CNN Model**
```bash
python train_cnn.py
```
#### **2. Train with VGG16 Transfer Learning**
```bash
python train_vgg16.py
```
#### **3. Train with ResNet50 Transfer Learning**
```bash
python train_resnet.py
```
#### **4. Real-time Webcam Detection**

```bash
python detect_mask_webcam.py --model best_model.h5
```

---

### **Folder / Project Structure**

```
real-time-mask-detection/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── notebooks/                # Optional: Jupyter notebooks
│   └── EDA.ipynb
│
├── models/
│   ├── cnn_model.h5
│   ├── vgg16_model.h5
│   └── resnet_model.h5
│
├── src/
│   ├── train_cnn.py
│   ├── train_vgg16.py
│   ├── train_resnet.py
│   └── detect_mask_webcam.py
│
├── requirements.txt
├── README.md
└── .gitignore
```
---
### **FastAPI Project Structure (Future Integration)**
```
mask_detection_api/
│
├── app/
│   ├── main.py           # FastAPI app entrypoint
│   ├── routes/
│   │   └── mask_detection.py
│   ├── models/
│   │   └── best_model.h5
│   └── utils/
│       └── predict.py    # Functions for prediction & preprocessing
│
├── venv/
├── requirements.txt
└── README.md
```
* `main.py` → Launches FastAPI server
* `mask_detection.py` → Endpoint for image/frame upload & mask prediction
* `predict.py` → Preprocessing + model inference logic
---

### **Future Enhancements**

* Add **helmet detection**, **social distancing monitoring**, or **crowd mask compliance**.
* Convert models to **TensorRT / ONNX** for real-time deployment on edge devices.
* Integrate with **React/Vue frontend** for dashboard display.
---
