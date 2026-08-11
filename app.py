import PIL.Image
import numpy as np
import pydicom  # <-- Added missing import
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image  # <-- Added missing import

NUM_CLASSES = 2  # Match the exact number of classes from your Kaggle notebook


@st.cache_resource
def load_cnn_model(num_classes):
  # 1. Recreate the exact Sequential wrapper architecture used during Kaggle training
  base_resnet = models.resnet50(weights=None)

  # Extract feature layers (everything except avgpool and fc)
  features = nn.Sequential(*list(base_resnet.children())[:-2])

  # Recreate the classifier block
  classifier = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(base_resnet.fc.in_features, 512),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(512, num_classes),
  )

  # Combine into a single module
  class CustomResNet(nn.Module):

    def __init__(self, feat, cls):
      super().__init__()
      self.features = feat
      self.classifier = cls

    def forward(self, x):
      x = self.features(x)
      x = self.classifier(x)
      return x

  model = CustomResNet(features, classifier)

  # 2. Load checkpoint dictionary
  checkpoint = torch.load(
      "resnet50_best.pt", map_location=torch.device("cpu"), weights_only=False
  )

  if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
  elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
  else:
    state_dict = checkpoint

  # Clean any module. prefixes if trained with DataParallel
  clean_state_dict = {
      k.replace("module.", ""): v for k, v in state_dict.items()
  }

  # 3. Load weights into the custom architecture
  model.load_state_dict(clean_state_dict)
  model.eval()
  return model


model = load_cnn_model(NUM_CLASSES)

# Optional: Define your readable class names (e.g., ["Normal", "Pneumonia"] or ["Benign", "Malignant"])
CLASS_NAMES = ["Class 0", "Class 1"]  # <-- Change these to your actual labels!

# File Uploader for images
uploaded_file = st.file_uploader(
    "Upload an image...", type=["dcm", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
  # Process DICOM files vs standard image files
  if uploaded_file.name.lower().endswith(".dcm"):
    # Read DICOM bytes
    dicom_data = pydicom.dcmread(uploaded_file)
    pixel_array = dicom_data.pixel_array.astype(float)

    # Rescale DICOM values (12-bit/16-bit) to standard 0-255 uint8 format
    pixel_array = (
        (pixel_array - pixel_array.min())
        / (pixel_array.max() - pixel_array.min() + 1e-8)
        * 255.0
    )
    pixel_array = pixel_array.astype(np.uint8)

    # Convert 2D grayscale array to a 3-channel RGB PIL Image
    image = Image.fromarray(pixel_array).convert("RGB")
  else:
    # Standard PNG/JPG handling
    image = Image.open(uploaded_file).convert("RGB")

  # Display in Streamlit
  st.image(image, caption="Uploaded Image", use_container_width=True)

  # Preprocess image to match PyTorch ResNet-50 standard inputs
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      ),
  ])

  img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

  # Predict
  if st.button("Classify Image"):
    with torch.no_grad():
      outputs = model(img_tensor)
      probabilities = torch.softmax(outputs, dim=1)
      _, predicted_class = torch.max(outputs, 1)

      class_id = predicted_class.item()
      confidence = probabilities[0][class_id].item() * 100

      st.success(f"**Prediction:** {CLASS_NAMES[class_id]}")
      st.info(f"**Confidence:** {confidence:.2f}%")