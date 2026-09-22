# ♻️ AI Waste Classification System

An AI-powered waste classification application built with **Python, TensorFlow, MobileNetV2, and Streamlit**.

The system classifies waste images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

If the highest prediction confidence is below the configured threshold, the result is displayed as **Unknown**.

---

## Features

- Image classification using MobileNetV2.
- Transfer learning with TensorFlow and Keras.
- Data augmentation for improved training.
- Fine-tuning support.
- Six waste categories.
- Confidence score display.
- Unknown result for low-confidence predictions.
- Streamlit web interface.
- Class probability visualization.
- Ready for future Arduino Nicla Vision integration.

---

## Project Workflow

```text
Upload Waste Image
        ↓
Resize Image to 224 × 224
        ↓
MobileNetV2 Preprocessing
        ↓
CNN Image Classification
        ↓
Calculate Confidence Score
        ↓
Waste Category or Unknown
```

---

## Waste Classes

The model is trained to classify the following categories:

```text
cardboard
glass
metal
paper
plastic
trash
```

The dataset folder must be organized as follows:

```text
dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Each folder should contain images belonging to the corresponding class.

---

## Unknown Prediction

The model is trained using six waste classes. It does not require a separate `unknown` folder.

Instead, the application uses a confidence threshold:

```python
CONFIDENCE_THRESHOLD = 0.70
```

The prediction rule is:

```text
Confidence >= 70%  →  Show predicted waste category
Confidence < 70%   →  Show Unknown
```

Example:

```text
plastic: 42%
paper: 25%
glass: 14%
metal: 10%
cardboard: 6%
trash: 3%
```

Final result:

```text
Unknown
```

---

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow
- Matplotlib
- Streamlit
- TrashNet Dataset

---

## Project Structure

```text
AI-Waste-Classification/
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── waste_classifier.keras
│   └── classes.txt
│
└── dataset-resized/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

Do not upload large datasets or model files to GitHub unless necessary. You can use `.gitignore` or Git Large File Storage.

---

## Dataset

This project uses the [TrashNet Dataset](https://github.com/garythung/trashnet).

TrashNet contains six waste categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

Download the dataset and place it inside the project directory.

---

## Installation

### Clone the repository

```bash
git clone [https://github.com/your-username/ai-waste-classification.git](https://github.com/your-username/ai-waste-classification.git)
cd ai-waste-classification
```

Replace `your-username` with your GitHub username.

### Create a virtual environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
tensorflow
numpy
pillow
matplotlib
streamlit
```

---

## Train the Model

Make sure the dataset is arranged correctly:

```text
dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Run the training script:

```bash
python train_model.py
```

The training process includes:

1. Loading the waste images.
2. Splitting the images into training and validation datasets.
3. Resizing images to \(224 \times 224\).
4. Applying image augmentation.
5. Loading the pretrained MobileNetV2 model.
6. Training the classification layer.
7. Fine-tuning selected MobileNetV2 layers.
8. Saving the trained model.
9. Saving the class names.

The trained model is saved in:

```text
models/waste_classifier.keras
```

---

## Run Image Prediction

Update the image path in `predict.py` and run:

```bash
python predict.py
```

Example output:

```text
--------------------------------
AI WASTE CLASSIFICATION
--------------------------------
Prediction : plastic
Confidence : 87.42%
```

Low-confidence example:

```text
--------------------------------
AI WASTE CLASSIFICATION
--------------------------------
Prediction : unknown
Confidence : 43.18%
```

---

## Run the Streamlit Application

Start the application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal. It is usually:

```text
http://localhost:8501
```

Upload an image to view:

- Uploaded image.
- Final prediction.
- Confidence score.
- Raw model prediction.
- Probability of every class.
- Unknown result for low-confidence predictions.

---

## Model Architecture

```text
Input Image: 224 × 224 × 3
        ↓
Data Augmentation
        ↓
MobileNetV2
        ↓
Global Average Pooling
        ↓
Dropout
        ↓
Dense Softmax Layer
        ↓
Six Waste Classes
```

MobileNetV2 is initially used as a frozen feature extractor. The final layers are then fine-tuned using a low learning rate.

---

## Configuration

The confidence threshold can be changed in `app.py`:

```python
CONFIDENCE_THRESHOLD = 0.70
```

Examples:

```python
CONFIDENCE_THRESHOLD = 0.60
```

```python
CONFIDENCE_THRESHOLD = 0.80
```

A higher threshold produces more `unknown` results. A lower threshold accepts more predictions but can increase incorrect classifications.

---

## Example Results

| Input image | Confidence | Final result |
|---|---:|---|
| Paper sheet | 94% | Paper |
| Plastic bottle | 89% | Plastic |
| Metal can | 91% | Metal |
| Phone | 43% | Unknown |
| Book | 52% | Unknown |
| Unclear waste image | 61% | Unknown |

The actual result depends on image quality, lighting, camera angle, and model performance.

---

## Limitations

- Paper and cardboard can be visually similar.
- Glass and plastic may sometimes be confused.
- TrashNet images may have simpler backgrounds than real-world waste.
- The confidence threshold cannot prevent every confident incorrect prediction.
- Performance may decrease with poor lighting or blurry images.
- Validation accuracy may differ from real-world accuracy.
- The model should be tested using new images captured with the final camera.

---

## Future Improvements

- Capture additional images using the Arduino Nicla Vision.
- Add images from different backgrounds and lighting conditions.
- Improve performance for weak classes.
- Add a separate waste-versus-non-waste classifier.
- Deploy the model on the Arduino Nicla Vision.
- Add real-time camera classification.
- Add a waste-counting dashboard.
- Store prediction history in a database.
- Add automatic waste sorting hardware.
- Convert the model to TensorFlow Lite or Edge Impulse format.

---

## Hardware Integration Plan

The planned hardware workflow is:

```text
Waste Object
        ↓
Arduino Nicla Vision Camera
        ↓
Image Capture
        ↓
Embedded AI Model
        ↓
Waste Classification
        ↓
Confidence Threshold
        ↓
Waste Category or Unknown
```

The current version runs through a Streamlit application. The model can later be optimized and deployed to an edge device.

---

## License

This project is intended for educational and prototype purposes.

The TrashNet dataset belongs to its original creator. Refer to the original dataset repository for its terms of use:

[TrashNet Dataset](https://github.com/garythung/trashnet)

---

## Author

**Pradeep Kumar Singha**

Nirmaan AI/ML Trainee 
Btech Final Year
