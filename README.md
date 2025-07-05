# NEURAL-STYLE-TRANSFER

company: CODTECH IT SOLUTIONS

name: MALLELA POOJA

intern id: CT2MTDL1319

domain: Artificial Intelligence

duration: 8 weeks

mentor: Neela Santosh

The Neural Style Transfer (NST) project is an exciting implementation of deep learning that allows us to generate artwork by combining the content of one image with the artistic style of another. This project leverages the power of Convolutional Neural Networks (CNNs) to blend a photograph (content image) with the aesthetics of an artwork (style image), resulting in a stylized image that looks like a painting created by an artist using a modern or classical style.

At the heart of this project is the use of pre-trained models, specifically the VGG19 convolutional neural network, which was trained on the ImageNet dataset. The model is not trained from scratch but used to extract features from both the content and style images. VGG19's layers are used to capture both the content representation (from deeper layers) and the style representation (from shallower layers) of the images.

The system follows a well-structured pipeline:

Image Preprocessing – The content and style images are loaded, resized, and transformed into tensors suitable for the model.

Feature Extraction – Using VGG19, specific layers are used to extract features that represent content (such as object structure) and style (such as brush strokes and color patterns).

Loss Functions – Two loss functions are used:

Content Loss measures how similar the generated image is to the content image.

Style Loss is calculated using Gram Matrices, which capture the correlation between different feature maps to represent the style.

Optimization – The system starts with a copy of the content image and uses the L-BFGS optimizer to iteratively update the pixels in the image. The goal is to minimize the combined loss (weighted sum of content and style losses) so that the output image gradually adopts the style of the style image while preserving the content structure.

Postprocessing and Output – After optimization, the image tensor is converted back to a human-readable image and displayed using matplotlib.

The deliverable is a Python script or Jupyter notebook that performs this entire process end-to-end. The output is a beautiful, stylized image that looks as if the content photo was painted in the style of the chosen artwork.

This project demonstrates how deep learning models can be repurposed for creative tasks. NST has practical applications in art generation, mobile photography filters, movie animation stylization, and more. It also serves as a great educational tool to understand how CNNs work internally, especially in feature extraction and transfer learning.

The script is written in Python using libraries such as PyTorch, Torchvision, PIL, and Matplotlib. The modularity of the code makes it easy to extend — for instance, adding options for multiple style layers, custom weight adjustments, or GUI/web integration for user uploads.

In conclusion, this project showcases the intersection of art and artificial intelligence, proving that neural networks can be used not only for solving analytical problems but also for inspiring creativity and transforming the way we interact with digital media.

#output
