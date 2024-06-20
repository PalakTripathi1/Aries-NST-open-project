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