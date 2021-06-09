import torch
from utils import load_img, gram_matrix
from models import vgg19
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_content_loss(gen_feat, orig_feat):
    # calculating the content loss of each layer by calculating the MSE between the content and generated features
    # and adding it to content loss
    content_l = torch.mean((gen_feat - orig_feat) ** 2)
    return content_l


def calc_style_loss(gen, style):
    batch_size, channel, height, width = gen.shape
    # Calculating the gram matrix for the style and the generated image
    G = gram_matrix(gen)
    A = gram_matrix(style)

    # Calcultating the style loss of each layer by calculating the MSE between the gram matrix of the style image and
    # the generated image and adding it to style loss
    style_l = torch.mean((G - A) ** 2)
    return style_l


def train():
    # the image optimization loop
    losses = []
    for epoch in range(epochs):

        # extracting the features of generated, content and the original required for calculating the loss
        gen_features = model(generated_image)
        orig_feautes = model(content_image)
        style_featues = model(style_image)

        # iterating over the activation of each layer and calculate the loss and add it to the
        # content and the style loss
        total_loss = calculate_loss(gen_features, orig_feautes, style_featues)
        # optimize the pixel values of the generated image and backpropagate the loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item()}')
            img = torch.clamp(generated_image[0], 0, 1)
            img = img.permute(1, 2, 0).cpu().detach().numpy()

            plt.imshow(img)
            plt.title(f'Epoch {epoch + 1}')
            plt.show()

        losses.append(total_loss.item())

    return losses


def calculate_loss(gen_features, orig_feautes, style_featues):
    style_loss = content_loss = 0
    for gen, cont in zip(gen_features[0], orig_feautes[0]):
        content_loss += calc_content_loss(gen, cont)

    for gen, style in zip(gen_features[1], style_featues[1]):
        style_loss += calc_style_loss(gen, style)

    # calculating the total loss - combination of the style loss and the content loss
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss


if __name__ == '__main__':
    # load  the style image and create the model

    style_image_path = 'data/styles/starrynight.jpg'
    style_image = load_img(style_image_path).to(device)

    # the output layers of of the vgg model
    style_layers_idx = [0, 5, 10, 19, 28]
    content_layers_idx = [19]

    model = vgg19(content_layers_idx, style_layers_idx).to(device)
    model = model.eval()

    epochs = 4000
    lr = 0.04
    # alpha represent how much we want to preserve the original content of the image
    alpha = 2
    # beta represent how much we style we want to add to the content image
    beta = 60

    content_image_path = 'data/content/elephant.jpg'
    content_image = load_img(content_image_path).to(device)

    # for faster calculation, the generated image initial values will be set to be the content image
    generated_image = content_image.clone().requires_grad_(True)

    # only the generate images values will be changed in the backprop process
    optimizer = torch.optim.Adam([generated_image], lr=lr)

    losses = train()

    # plot the losses per epoch
    plt.plot(losses)
    plt.title("Loss per epoch")
    plt.show()
