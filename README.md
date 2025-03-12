# Utilizing Convolutional Neural Networks for Calculating Full-Field Stress Components and Directions in Photoelasticity

This repository contains the source code for the paper *"Utilizing Convolutional Neural Networks for Calculating Full-Field Stress Components and Directions in Photoelasticity."* The work proposes a novel method based on convolutional neural networks (CNNs) to directly obtain the principal stress components and directions from photoelastic data, overcoming the limitations of traditional methods that can only provide the principal stress difference (the difference between the first and second principal stresses).

## Overview

In this paper, a convolutional neural network is proposed for the first time, capable of directly and simultaneously determining the full-field first principal stress, second principal stress, and principal stress direction. A dataset generation method was developed to train this network, producing a novel high-quality dataset containing over 41,000 raw samples, without data augmentation. The proposed network exhibits high accuracy and strong generalization across synthetic and experimental validation sets. On the synthesized set, the structural similarity exceeds 0.98, and the mean squared error is below 0.45, with similarly satisfactory results on the real validation set. This network establishes a direct link between photoelastic images and full-field stress components and directions, enhancing the efficiency and capability of photoelasticity and broadening its application range. The developed dataset generation method may offer helpful insights for further advancement of deep learning in the field of photoelasticity.

## Highlights

1.	Proposed a CNN capable of directly determining the full-field principal stress components and directions.
2.	Developed a method to convert stress information from finite element software into sample datasets.
3.	Created a dataset featuring photoelastic images as inputs and the corresponding full-field stress components and directions as labels.
4.	Enhanced the efficiency and capability of the photoelastic method, expanding its range of applications.

# dataset
*The dataset can be accessed on Mendeley Data via DOI: 10.17632/fkyv6r752g.1.*

## Requirements

To run this project, you will need the following libraries:

- Python 3.9
- TensorFlow or PyTorch (depending on your implementation choice)


## Acknowledgments

- Special thanks to the contributors of the libraries used in this project.
- Thanks to the researchers and collaborators in the field of digital photoelasticity for their valuable insights.
