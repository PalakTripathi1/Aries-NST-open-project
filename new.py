%matplotlib inline
# Download the images using wget command
!wget"C:\Users\tripa\Downloads\stars.jpg"
!wget"C:\Users\tripa\Downloads\Princess.jpg"
from PIL import Image
import matplotlib.pyplot as plots
import numpy as py

import torch
import torch.optim as op
from torchvision import transforms, models
# get the "features" portion of VGG19 (we will not need the "classifier" portion)
vg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for p in vg.parameters():
    p.requires_grad_(False)
    # move the model to GPU, if available
twice = torch.device("c" if torch.cuda.is_available() else "useCPU")

vg.to(twice)
def image_processing(path, sz=500, image_shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 500 pixels in the x-y dims.'''
    
    picture = Image.open(path).convert('RGB')
    
    # large images will slow down processing
    if max(picture.size) > sz:
        size= sz
    else:
        size = max(picture.size)
    
    if image_shape is not None:
        size = image_shape
    
    in_transform = transforms.Compose([ transforms.Resize((size, size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))])
    
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    picture = in_transform(picture)[:3, :, :].unsqueeze(0)
    
    return picture
    def converting(tensor):
    """ Display a tensor as an image. """
    
    picture = tensor.to("useCPU").clone().detach()
    picture = picture.numpy().squeeze()
    picture = picture.transpose(1, 2, 0)
    picture = picture * py.array((0.229, 0.224, 0.225)) + py.array((0.485, 0.456, 0.406))
    picture = picture.clip(0, 1)

    return picture
    # Load images
img1_to_be_modified = load_image('Princess.jpg').to(twice)
styles = load_image('stars.jpg', shape=img1_to_be_modified.shape[-2:]).to(twice)

# Display images
fig, (ax1, ax2) = plots.subplots(1, 2, figsize=(20, 10))
ax1.imshow(converting(img1_to_be_modified))
ax1.set_title('Image To Be Modified')
ax2.imshow(im_convert(styles))
ax2.set_title('Styling Image')
plots.show()
# Define content and style loss functions, gram matrix function, and the training loop

# Function to calculate the gram matrix
def gramgrid(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram 

# Define content loss
def img_loss(target, content):
    return torch.mean((target-content)**2)

# Define style loss
def styles_loss(target, style):
    targetg= gramgrid(target)
    styleg= gramgrid(style)
    return torch.mean((targetg-styleg)**2)

# Extract features from the images using the VGG model
def features(image, model, layering=None):
    if layering is None:
        layering = {
            '0': 'conv1_1',
            '5': 'conv2_1'
            '11': 'conv3_1'
            '21': 'conv4_2',  # content layer
            
        }
    features2 = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layering:
            features[layering[name]] = x
    return features

# Get the features of the content and style images
content_features = get_features(img1_to_be_modified, vg)
style_features = get_features(styles, vg)

# Calculate the gram matrices for the style features
style_grams = {layer: gramgrid(style_features[layer]) for layer in style_features}

# Create a target image and make it require gradients
target = content_image.clone().requires_grad_(True).to(twice)

# Set weights for each style layer
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

content_weight = 1  # alpha
style_weight = 1e6  # beta

# Set up the optimizer
optimizing = op.Adam([target], lr=0.003)

# Training loop
for epoch in range(1, 2001):
    targetfe= get_features(target, vg)
    img_losses = img_loss(targetfe['conv4_2'], content_features['conv4_2'])

    style_loss = 0
    for layer in style_weights:
        targetfe= targetfe[layer]
        style_gram = style_grams[layer]
        _, d, h, w = targetfe.shape
        target_gram = gram_matrix(targetfe)
        layerloss = style_weights[layer] * styles_loss(target_gram, style_gram)
        style_loss += layerloss / (d * h * w)

    total_loss = content_weight * img_losses + style_weight *style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 400 == 0:
        print('Epoch {}: Total Loss: {:.4f}'.format(epoch, total_loss.item()))

# Display the final stylized image
final_img = converting(target)
plots.figure(figsize=(10, 10))
plots.imshow(final_img)
plots.title('Stylized Image')
plots.show()