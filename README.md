# Senior-Project

This senior project is meant to serve as a website to train a deep learning model
to detect simple defects in parts. Once the model has been trained, a user is able to evaluate the model in order to determine it's effectiveness.
Once the user is satisfied with the evaluation of the model, they are able to download the model. A user is also able to
connect to a camera in order to see real time defect detections in action. 

In order to make this website functional, Django was used as the web framework. The Deep Learning model used is a Convolutionary Neural Network paired
with data augmentation in the cases of limited part images. In order to allow the system to train with keras, tensorflow, cuDNN SDK, and the CUDA toolkit must be installed
on the users machine.
