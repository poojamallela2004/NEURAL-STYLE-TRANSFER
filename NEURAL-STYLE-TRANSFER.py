pip install torch torchvision matplotlib pillow

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Function to load and preprocess images
def image_loader(image_path, max_size=400):
    image = Image.open(image_path)
    size = max(max(image.size), max_size)
    loader = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Function to show the tensor image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load content and style images
content_img = image_loader("content.jpg")
style_img = image_loader("style.jpg")

# Display the input images
imshow(content_img, title="Content Image")
imshow(style_img, title="Style Image")

# Use a pre-trained VGG19 model
cnn = models.vgg19(pretrained=True).features.to(device).eval()

# Loss functions
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Normalization
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
    def forward(self, x):
        return (x - self.mean) / self.std

# Create model
def get_style_model_and_losses(cnn, style_img, content_img):
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential(Normalization(normalization_mean, normalization_std)).to(device)
    content_losses, style_losses = [], []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:i+1]

    return model, style_losses, content_losses

# Run style transfer
input_img = content_img.clone()
optimizer = optim.LBFGS([input_img.requires_grad_()])
model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

# Optimization loop
print("Optimizing...")
run = [0]
while run[0] <= 300:
    def closure():
        input_img.data.clamp_(0, 1)
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_score * 1e6 + content_score
        loss.backward()
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Iteration {run[0]}: Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")
        return loss

    optimizer.step(closure)

# Final output
input_img.data.clamp_(0, 1)
imshow(input_img, title="Stylized Image")
