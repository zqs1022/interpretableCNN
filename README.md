# Interpretable Convolutional Neural Networks


# Introduction

This paper proposes a method to modify traditional convolutional neural networks (CNNs) into interpretable CNNs, in order to clarify knowledge representations in high conv-layers of CNNs. In an interpretable CNN, each filter in a high conv-layer represents a certain object part. We do not need any annotations of object parts or textures to supervise the learning process. Instead, the interpretable CNN automatically assigns each filter in a high conv-layer with an object part during the learning process. Our method can be applied to different types of CNNs with different structures. The clear knowledge representation in an interpretable CNN can help people understand the logics inside a CNN, i.e., based on which patterns the CNN makes the decision. Experiments showed that filters in an interpretable CNN were more semantically meaningful than those in traditional CNNs.

# Citation

Please cite the following paper, if you use this code.
Quanshi Zhang, Ying Nian Wu, and Song-Chun Zhu, "Interpretable Convolutional Neural Networks," in CVPR 2018

# Code

We released the code with slight technical extensions to the above paper for more robustness. For example, the code learned the parameter \beta instead of simply setting \beta=4.

We will release the code based on PyTorch and TensorFlow, later.

# How to use

run demo.m

Note that please set in the window of the MATLAB following "HOME --> Preferences --> MATLAB --> General --> MAT-Files --> MATLAB Version 7.3 or later." Thus, the Matlab can save large MAT files.

Please see demo.m for detailed introduction of the code
