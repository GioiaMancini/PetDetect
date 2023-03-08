# PetDetect - Android App to recognize your pet through AI :cat: :dog: :mag:
Authors: Christian Brignone, Gioia Mancini

(One day projects)


This Android app allows the user to recognize each of his/her pets.

The app involves the following features:
* Select an image from local storage
* Capture an image through the camera
* Identify generic objects in the selected/captured scene
* Recognize the identity of a pet

In the following, we show an example of the main functionalities with the partecipation of our two cats: Willy (white and grey tiger cat) and Ombra (brown tabby cat).

![](https://github.com/ChristianBrignone/PetDetect/blob/master/petDetect_demo.gif)

The app uses a TensorFlow Lite MobileNet model to identify generic objects. 
In order to personalize the functionalities and make MobileNet able to recognize the identity of a specific pet, we fine-tuned the neural network on an ad-hoc dataset, containing the images of our pets.
