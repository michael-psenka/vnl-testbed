import random

import torch
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms

import jax.numpy as jnp

# important lines
# *---------------------------------------------------------*

# 83: Start of main text prompt switch case
# 188: FINAL PROCESSING: placing sub-images into the main image

# *---------------------------------------------------------*

# construct a dataset consisting of multiple MNIST digits in an image, with a text
# description of the digits. this will first randomly select a promt, then generate
# the image from the prompt's description.

# dataset will have num_samples images, omitting digits in the classes_omit list of
# positive integers, and omitting text in the text_omit list of strings

# returns a tensor of shape (num_samples, 3, 100, 100) containing the images, and a
# list of strings of size num_samples containing the text descriptions
def generate(num_samples=10000,export_jax=False, classes_omit=[], text_omit=[]):

    # text takes the form of a base image followed by a sequence of modifications:
    # "a [base_iamge], [mod_1], [mod_2], ..., [mod_n]."
    
    # below is a list of all possible modification

    text_templates = [
        ".", # need a choice to stop the description early
        "to the left of a *", # a star can be replaced with a digit's class
        "to the right of a *",
        "above a *",
        "below a *",
        "the _ colored &", # _ is the original chosen digit, & can be a color
    ]

    # possible colors, represented as a dict with keys as color names (string) and
    # the values to multiplicitavely scale down a grayscale image
    colors = {
        "red": (1, 0.1, 0.1),
        "green": (0.1, 1, 0.1),
        "blue": (0.1, 0.1, 1),
        "yellow": (1, 1, 0.1),
        "purple": (0.5, 0.1, 0.5),
    }

    # load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                shuffle=True, num_workers=2)
    # datak iterator
    dataiter = iter(trainloader)

    # sample an image and get its label
    image, label = next(dataiter)
    # convert image into a color image, where each channel is the grayscale image
    image = torch.cat((image, image, image), dim=1)
    # default coordinates of image will be around the center
    image_coord = (random.randint(40, 60), random.randint(40, 60))
    # holder for new image to append alongside original
    image_new = None


    images_full = torch.zeros(num_samples, 3, 100, 100)

    print('Creating MultiMNIST dataset...')
    for i in tqdm(range(num_samples)):

        # INNER LOOP: keep randomly choosing components to build up the text description
        text_desc = f"A {label.item()}"
        remaining_templates = text_templates.copy()


        while len(remaining_templates) > 0:
            # choose a random template
            template = remaining_templates.pop(random.randint(0, len(remaining_templates) - 1))
            # if we chose period, break
            if template == ".":
                text_desc += "."
                break

            # Start of main text prompt switch case
            # now, make image modification based on text description
            if template == "to the left of a *":
                # replace the * with a random digit (not the originally chosen digit)
                for i in range(100):
                    image_new, label_new = next(dataiter)
                    if label_new.item() != label.item():
                        # replace the * in the template with the new digit
                        template = template.replace("*", str(label_new.item()))
                        break
                    elif i == 99:
                        # return error that somehow couldn't find different digit (probability 1/10^100)
                        raise Exception("A different digit could not be randomly chosen (should be practically impossible, probability 1/10^100).")
                    
                # set coordinates of the original and new image
                image_coord = (random.randint(15, 35), random.randint(40, 60))
                image_new_coord = (random.randint(65, 85), random.randint(40, 60))

                # remove all other positional templates from being chosen
                remaining_templates.remove("to the right of a *")
                remaining_templates.remove("above a *")
                remaining_templates.remove("below a *")

            if template == "to the right of a *":
                # replace the * with a random digit (not the originally chosen digit)
                for i in range(100):
                    image_new, label_new = next(dataiter)
                    if label_new.item() != label.item():
                        # replace the * in the template with the new digit
                        template = template.replace("*", str(label_new.item()))
                        break
                    elif i == 99:
                        # return error that somehow couldn't find different digit (probability 1/10^100)
                        raise Exception("A different digit could not be randomly chosen (should be practically impossible, probability 1/10^100).")
                    
                # set coordinates of the original and new image
                image_coord = (random.randint(65, 85), random.randint(40, 60))
                image_new_coord = (random.randint(15, 35), random.randint(40, 60))

                # remove all other positional templates from being chosen, including this one
                remaining_templates.remove("to the left of a *")
                remaining_templates.remove("above a *")
                remaining_templates.remove("below a *")
                    
            if template == "above a *":
                # replace the * with a random digit (not the originally chosen digit)
                for i in range(100):
                    image_new, label_new = next(dataiter)
                    if label_new.item() != label.item():
                        # replace the * in the template with the new digit
                        template = template.replace("*", str(label_new.item()))
                        break
                    elif i == 99:
                        # return error that somehow couldn't find different digit (probability 1/10^100)
                        raise Exception("A different digit could not be randomly chosen (should be practically impossible, probability 1/10^100).")
                    
                # set coordinates of the original and new image
                image_coord = (random.randint(40, 60), random.randint(15, 35))
                image_new_coord = (random.randint(40, 60), random.randint(65, 85))

                # remove all other positional templates from being chosen, including this one
                remaining_templates.remove("to the left of a *")
                remaining_templates.remove("to the right of a *")
                remaining_templates.remove("below a *")
                
            if template == "below a *":
                # replace the * with a random digit (not the originally chosen digit)
                for i in range(100):
                    image_new, label_new = next(dataiter)
                    if label_new.item() != label.item():
                        # replace the * in the template with the new digit
                        template = template.replace("*", str(label_new.item()))
                        break
                    elif i == 99:
                        # return error that somehow couldn't find different digit (probability 1/10^100)
                        raise Exception("A different digit could not be randomly chosen (should be practically impossible, probability 1/10^100).")
                    
                # set coordinates of the original and new image
                image_coord = (random.randint(40, 60), random.randint(65, 85))
                image_new_coord = (random.randint(40, 60), random.randint(15, 35))

                # remove all other positional templates from being chosen, including this one
                remaining_templates.remove("to the left of a *")
                remaining_templates.remove("to the right of a *")
                remaining_templates.remove("above a *")
                
            
            if template == "the _ colored &":
                # first, replace _ with the original image label
                template = template.replace("_", str(label.item()))
                # next, replace & with a random color
                color = random.choice(list(colors.keys()))
                template = template.replace("&", color)
                # get the hue for the template, and initialize a colorjitter from the hue
                scaling_factors = colors[color]
                # finally, scale image to get colored image
                image[:,0,:,:] = image[:,0,:,:] * scaling_factors[0]
                image[:,1,:,:] = image[:,1,:,:] * scaling_factors[1]
                image[:,2,:,:] = image[:,2,:,:] * scaling_factors[2]

            # add template to description
            text_desc += ", " + template


        # FINAL PROCESSING: placing sub-images into the main image
        # create a black color (100,100) image in pytorch format
        image_final = torch.zeros(3, 100, 100)

        # add the original image into image_final such that its center is at image_coord
        # NOTE: coordinates are given in (x, y), which we need to convert to (row, col)
        image_final[:, image_coord[1] - 14:image_coord[1] + 14, image_coord[0] - 14:image_coord[0] + 14] = image
        # if image_new is defined, add it to image_final such that its center is at image_new_coord
        if image_new != None:
            image_final[:, image_new_coord[1] - 14:image_new_coord[1] + 14, image_new_coord[0] - 14:image_new_coord[0] + 14] = image_new

        images_full[i,:,:,:] = image_final

    # if exporting for JAX, comvert to jax.numpy array
    if export_jax:
        images_full = jnp.array(images_full.numpy())
        # switch dimensions to (N, H, W, C)
        images_full = jnp.transpose(images_full, (0, 2, 3, 1))

    # finally, return image_final and text_desc
    return images_full, text_desc