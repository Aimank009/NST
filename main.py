from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
import torch
from torchvision.models import vgg19
from PIL import Image
import io
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation to tensor
image_size = 256
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()])


def load_image(image_data):
    image = Image.open(io.BytesIO(image_data))
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device, torch.float)


# Define content loss module
class ContentLoss(nn.Module):
    def __init__(self, target_feature):
        super(ContentLoss, self).__init__()
        self.target = target_feature.detach()  # Detach the target to prevent gradient computation

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


# Compute Gram matrix for style loss
def gram_matrix(input):
    batch_size, channels, height, width = input.size()
    features = input.view(batch_size * channels, height * width)
    G = torch.mm(features, features.t())
    return G.div(batch_size * channels * height * width)


# Define style loss module
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# Load the VGG-19 model
cnn = vgg19(pretrained=True).features.to(device).eval()

# Define normalization
normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# Set the layers for style and content losses
content_layer_names = ['conv_3']
style_layer_names = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# Build the model with style and content losses
def build_style_model_and_losses(cnn, mean, std, style_img, content_img,
                                 content_layers=content_layer_names, style_layers=style_layer_names):
    normalization = Normalization(mean, std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    layer_index = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            layer_index += 1
            name = f'conv_{layer_index}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{layer_index}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{layer_index}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{layer_index}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{layer_index}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f'style_loss_{layer_index}', style_loss)
            style_losses.append(style_loss)

    # Remove layers after the last style or content loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses


# Optimizer for input image
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_(True)])
    return optimizer


# Execute style transfer
def perform_style_transfer(cnn, mean, std, content_img, style_img, input_img, num_steps=100, style_weight=1000000,
                           content_weight=1):
    model, style_losses, content_losses = build_style_model_and_losses(cnn, mean, std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Starting style transfer...')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Iteration {run[0]}:")
                print(f'Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}')
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img


@app.route('/')
def home():
    return send_from_directory('.', 'templates/index.html')


@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    content_image = request.files['content']
    style_image = request.files['style']

    content_img = load_image(content_image.read())
    style_img = load_image(style_image.read())
    input_img = content_img.clone()

    output = perform_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img)

    output_image = transforms.ToPILImage()(output.squeeze(0).cpu().detach())
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name='stylized_image.png')


if __name__ == '__main__':
    app.run(debug=True)
