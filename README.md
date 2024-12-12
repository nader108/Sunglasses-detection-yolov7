# Sunglasses Detection using YOLOv7

Project Overview
This project leverages YOLOv7 (You Only Look Once), a state-of-the-art real-time object detection model, to identify sunglasses in images. The model is trained on a custom dataset consisting of three classes: glasses, no_glasses, and sun_glasses. The objective is to detect whether an individual is wearing sunglasses, glasses, or neither in real-time or on still images.

Key Features
Real-time Detection: The model is capable of detecting sunglasses in real-time with high accuracy.
YOLOv7 Architecture: The project uses YOLOv7, a highly efficient and fast object detection model for detecting objects in images.
Custom Dataset: The model is trained on a custom dataset consisting of images classified into three categories:
Glasses: Images with regular glasses.
No Glasses: Images without any eyewear.
Sun Glasses: Images with sunglasses.
Dataset
The dataset for training the model was sourced from Roboflow and contains labeled images for the three categories. You can access and explore the dataset here on Roboflow.

Technologies Used
YOLOv7: For efficient and real-time object detection.
PyTorch: The framework used for training the model.
ONNX: For exporting the model to a format that can be used in various platforms.
Google Colab: For training the model using free GPU resources.
OpenCV: For image preprocessing and augmentation.
Model Training
YOLOv7 Setup: The YOLOv7 model is cloned from the official repository and customized to detect sunglasses.
Dataset Preparation: The dataset is preprocessed and divided into training and testing sets.
Training the Model: The model was trained for 50 epochs using a batch size of 8 and an image size of 640x640.
Performance Metrics: The model achieved excellent performance with 99% accuracy on test data, and the mAP (Mean Average Precision) for the detection of sunglasses reached 0.79.
Model Evaluation
The trained model was evaluated using several metrics:

Precision: 98.1% for detecting sunglasses.
Recall: 99.6% for detecting sunglasses.
mAP: 98.5% (at IoU 0.5).
mAP at 0.5:.95: 79% (for multiple thresholds).
ONNX Export
After training, the model was exported to the ONNX format for broader compatibility. The ONNX model can be easily loaded and used for inference across various platforms.

How to Use
Clone the repository:

bash
Copy code
git clone https://github.com/nader108/sunglasses-detection-yolov7
cd sunglasses-detection-yolov7
Install required dependencies:

bash
Copy code
pip install -r requirements.txt
To run inference on an image, use the following command:

bash
Copy code
python detect.py --weights runs/train/sunglasses_dete4/weights/best.onnx --img-size 640 --source face-203.jpg
This will process the image face-203.jpg and detect if the person is wearing sunglasses.

To use the model in other platforms, you can load the ONNX model using:

python
Copy code
import onnx
import onnxruntime as ort

# Load the model
model_path = 'path/to/best.onnx'
ort_session = ort.InferenceSession(model_path)

# Run inference
outputs = ort_session.run(None, {'input': input_data})
Future Work
Improve Dataset: Adding more diverse images to improve model generalization.
Real-Time Video Detection: Deploy the model for real-time detection in live video streams.
Detect Other Accessories: Extend the model to detect other accessories like hats or masks.
Results
The trained YOLOv7 model is highly accurate and fast in detecting sunglasses, offering a reliable solution for sunglasses detection in images and videos.

