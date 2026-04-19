import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import torch.nn as nn

# --- CUSTOM ARCHITECTURES ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class CBAMResNet50(nn.Module):
    def __init__(self, num_classes=5):
        super(CBAMResNet50, self).__init__()
        try:
            resnet = models.resnet50(weights=None)
        except TypeError:
            resnet = models.resnet50(pretrained=False)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.cbam3 = CBAM(1024)
        self.layer4 = resnet.layer4
        self.cbam4 = CBAM(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- CONFIGURATION ---
# Define the target image size expected by your model
TARGET_SIZE = (224, 224) 

# Define your class labels here. Modify these based on your specific SIPaKMeD training
CLASS_LABELS = [
    'ImDys (Immature Dysplastic)', 
    'Koil (Koilocytotic)', 
    'Meta (Metaplastic)', 
    'Parab (Parabasal)', 
    'Super (Superficial-Intermediate)'
]

# Define the name of the last convolutional layer for Grad-CAM
# E.g., 'features.7' for a simple CNN, 'layer4.2.conv3' for ResNet50
DEFAULT_LAST_CONV_LAYER = 'features'

st.set_page_config(page_title="SIPaKMeD Cell Classifier", page_icon="🔬", layout="wide")

# --- FUNCTIONS ---

@st.cache_resource
def load_trained_model(model_path):
    """Loads the trained PyTorch model. Cached to avoid reloading."""
    try:
        model_data = torch.load(model_path, map_location=torch.device('cpu'))
        
        if isinstance(model_data, dict) or not hasattr(model_data, 'eval'):
            st.info("Detected a PyTorch 'state_dict'. Analyzing architecture...")
            
            # Extract actual state_dict if it's nested (e.g. {'model_state_dict': ...})
            if 'model_state_dict' in model_data:
                state_dict = model_data['model_state_dict']
            elif 'state_dict' in model_data:
                state_dict = model_data['state_dict']
            else:
                state_dict = model_data
                
            # Determine architecture based on keys
            keys = list(state_dict.keys())
            has_cbam = any('cbam' in k for k in keys)
            has_fc1_fc2 = 'fc1.weight' in keys and 'fc2.weight' in keys
            
            if has_cbam and has_fc1_fc2:
                st.success("Detected custom CBAM-ResNet50 architecture!")
                model = CBAMResNet50(num_classes=len(CLASS_LABELS))
            else:
                st.info("Falling back to standard ResNet50 architecture.")
                try:
                    model = models.resnet50(weights=None)
                except TypeError:
                    model = models.resnet50(pretrained=False)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, len(CLASS_LABELS))
            
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                st.error(f"Failed to load weights. Architecture mismatch. Error details: {e}")
                return None
        else:
            model = model_data

        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Note: PyTorch models saved as 'state_dict' require the correct architecture to load into.")
        return None

def preprocess_image(img, target_size):
    """Preprocesses the image for the PyTorch model."""
    # Define standard PyTorch transformations
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        # Normalize using ImageNet standards (adjust if you used different normalization)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
    return img_tensor

def get_layer_by_name(model, layer_name):
    """Helper to get a PyTorch layer by its string name."""
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None

class GradCAM:
    """PyTorch Grad-CAM implementation"""
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.target_layer = get_layer_by_name(model, target_layer_name)
        
        self.gradients = None
        self.activations = None
        
        if self.target_layer is not None:
            self.target_layer.register_forward_hook(self.save_activation)
            self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        if self.target_layer is None:
            return None, None
            
        self.model.zero_grad()
        
        # Forward pass
        preds = self.model(x)
        
        # Get probabilities using Softmax
        probs = F.softmax(preds, dim=1)[0]
        
        if class_idx is None:
            class_idx = preds.argmax(dim=1).item()
        
        # Backward pass
        score = preds[0, class_idx]
        score.backward()
        
        # Calculate Grad-CAM
        if self.gradients is None or self.activations is None:
            return None, probs.detach().cpu().numpy()
            
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight activations
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (x.shape[3], x.shape[2]))
        
        if np.max(heatmap) == 0:
            return heatmap, probs.detach().cpu().numpy()
            
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        
        return heatmap, probs.detach().cpu().numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlays the heatmap on the original image."""
    img = np.array(img.resize(TARGET_SIZE))
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap] * 255.0

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    
    # Ensure values are within valid range and cast to uint8
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img


# --- UI LAYOUT ---

# Sidebar
with st.sidebar:
    st.markdown("## 🔬 AI Health Demo")
    st.title("About the Project")
    st.write(
        "This tool uses a Convolutional Neural Network (CNN) to classify cervical cells "
        "from the **SIPaKMeD dataset** into 5 distinct categories."
    )
    st.markdown("---")
    st.subheader("Dataset (SIPaKMeD)")
    st.write(
        "The SIPaKMeD database consists of cropped images of isolated cells extracted from Pap smear slides. "
        "Classes include:"
    )
    for label in CLASS_LABELS:
        st.markdown(f"- {label}")
    
    st.markdown("---")
    st.subheader("Model Information")
    st.write("- **Architecture:** PyTorch CNN")
    st.write("- **Explainability:** Grad-CAM")
    
    st.markdown("---")
    st.subheader("Instructions")
    st.write("1. Upload your `.pt` model file.")
    st.write("2. Upload a cervical cell image (`.jpg` or `.png`).")
    st.write("3. Review the prediction, confidence distribution, and Grad-CAM visualization.")


# Main App Header
st.title("🔬 Deep Learning Interpretability: Cervical Cell Classification")
st.markdown("Upload a cell image to predict its class and visualize the PyTorch model's focus using **Grad-CAM**.")

# Setup Model Upload/Path
st.markdown("### 1. Model Setup")

uploaded_model = st.file_uploader("Upload PyTorch Model (.pt or .pth)", type=["pt", "pth"])
model = None
last_conv_layer_name = DEFAULT_LAST_CONV_LAYER

if uploaded_model is not None:
    # Save the uploaded model temporarily to load it
    with open("temp_model.pt", "wb") as f:
        f.write(uploaded_model.getbuffer())
    with st.spinner("Loading PyTorch model..."):
        model = load_trained_model("temp_model.pt")
        if model:
            st.success("Model loaded successfully!")
            
            # Extract potential conv layer names to help the user
            conv_layers = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    conv_layers.append(name)
            
            if conv_layers:
                suggested_layer = conv_layers[-1]
                last_conv_layer_name = st.text_input("Last Conv Layer Name (for Grad-CAM):", value=suggested_layer)
                st.caption(f"Detected PyTorch convolutional layers. Using `{suggested_layer}` by default.")
            else:
                last_conv_layer_name = st.text_input("Last Conv Layer Name (for Grad-CAM):", value=DEFAULT_LAST_CONV_LAYER)
                st.warning("Could not automatically detect a Conv2d layer. Please enter it manually based on your model's `named_modules()`.")
else:
    st.warning("Please upload a trained `.pt` model to proceed.")


st.markdown("---")
st.markdown("### 2. Image Input & Prediction")

if model is not None:
    uploaded_file = st.file_uploader("Upload Cell Image", type=["jpg", "png", "jpeg", "bmp", "dat"])

    if uploaded_file is not None:
        # Handle the common SIPaKMeD mistake of uploading coordinate .dat files
        if uploaded_file.name.endswith(".dat"):
            try:
                # Try to read it as an image just in case it's a renamed image
                image = Image.open(uploaded_file).convert('RGB')
            except Exception:
                st.error(f"❌ You uploaded `{uploaded_file.name}`, which is a text file containing coordinates, NOT an image.")
                st.info("💡 **Hint:** In the SIPaKMeD dataset, `.dat` files contain cell boundary coordinates. The actual cell images are the **`.bmp`** files (e.g., `001_01.bmp` or `001.bmp`). Please upload the corresponding `.bmp` image file instead!")
                st.stop()
        else:
            try:
                image = Image.open(uploaded_file).convert('RGB')
            except Exception as e:
                st.error("Error loading image. Please ensure it's a valid image file.")
                st.stop()

        # Layout columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption='Original Image', use_container_width=True)
            
        with st.spinner("Analyzing image..."):
            # Preprocess
            preprocessed_img = preprocess_image(image, TARGET_SIZE)
            
            # Predict & GradCAM
            grad_cam = GradCAM(model, last_conv_layer_name)
            heatmap, probabilities = grad_cam(preprocessed_img)
            
            if probabilities is not None:
                pred_class_index = np.argmax(probabilities)
                pred_class_name = CLASS_LABELS[pred_class_index]
                confidence = probabilities[pred_class_index] * 100
            
        with col2:
            st.subheader("Prediction Results")
            st.markdown(f"**Predicted Class:** <span style='color:#E63946; font-weight: bold; font-size: 24px;'>{pred_class_name}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** <span style='font-size: 20px; font-weight: bold;'>{confidence:.2f}%</span>", unsafe_allow_html=True)
            
            # Confidence Bar Chart
            st.markdown("#### Confidence Distribution")
            fig = px.bar(
                x=probabilities * 100,
                y=CLASS_LABELS,
                orientation='h',
                labels={'x': 'Confidence (%)', 'y': ''},
                color=probabilities,
                color_continuous_scale='Blues'
            )
            # Reorder y-axis to match the array order nicely
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                showlegend=False, 
                margin=dict(l=0, r=0, t=10, b=0), 
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 3. Explainability (Grad-CAM)")
        
        with st.spinner("Generating Grad-CAM Heatmap..."):
            if heatmap is not None:
                superimposed_img = overlay_heatmap(image, heatmap)
                
                gc_col1, gc_col2, gc_col3 = st.columns(3)
                with gc_col1:
                    st.image(image.resize(TARGET_SIZE), caption="Original Input", use_container_width=True)
                with gc_col2:
                    # Display heatmap
                    fig_heat, ax = plt.subplots()
                    ax.imshow(heatmap, cmap='jet')
                    ax.axis('off')
                    # Save plot to buffer to render in Streamlit without tight_layout warnings
                    st.pyplot(fig_heat, use_container_width=True, clear_figure=True)
                    st.caption("Grad-CAM Heatmap (Raw)")
                with gc_col3:
                    st.image(superimposed_img, caption="Superimposed Overlay", use_container_width=True)
                
                st.info("💡 **What does this mean?** The heatmap highlights the regions of the cell image that most heavily influenced the model's prediction. Red/yellow areas indicate high importance, while blue areas indicate low importance.")
            else:
                st.error(f"Failed to generate Grad-CAM. Ensure the layer name `{last_conv_layer_name}` is correct and exists in the model.")
