# Apply GAN Model to Dog and Cat images from CIFAR-10 dataset

As part of the class project, I observed Generative Adversarial Networks (GANs) and saw how new images can be generated from random and fool the discriminator by learning from the losses. I followed the tutorial [here](https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/?fbclid=IwAR2FetsF4lEoXrPsiXLAmEFdyOxQFRxjhwerJouxbTCfjafMO8JwMFuw70g). 

As GANs take much computation power, I started from relatively smaller image sizes, and cifar10 images meet my desirability by having images with dimension 32 by 32 pixels. I only selected class dog (500 images) to train the network, again because of the heavy computation needed to produce results.

*Here are some dog images in the CIFAR-10 dataset.*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159186725-226cd3b8-5e76-4b34-9d80-17ef0eb3b7f0.png" alt="Some dog images of CIFAR-10" width="600px"/>
</p>

The idea of Generative Adversarial Networks (GANs) is that you have a discriminator model that tries to learn to differentiate real images and fake images, which is a binary classification problem. The real images are legit images, generated from real life for example the CIFAR-10 dataset. The fake images are generated from the generator model that tries to fool the discriminator and tries to make the discriminator that it is just the same as a real image. 

## Architecture

Here are the layers for the discriminator. The input is the image with three color channels for each pixel and a total of 32x32 pixels. I also used stride, leaky ReLU, dropout, and the Adam optimizer for the discriminator.   

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159187106-c65a9dee-fcce-49da-9708-e6bb64866394.png" alt="discriminator function" width="600px"/>
</p>

Here is the generator used to fool the discriminator. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159187148-06808c99-d4f7-4252-83ca-e27fc9e2d05f.png" alt="generator function" width="600px"/>
</p>

I initialized the generator with values randomly drawn from the Gaussian distribution. I also used strides and leaky ReLU like above, but unlike many neural networks going from a high dimension to a lower one, the generator has to output 32*32*3 = 3,072 values from the number of Gaussian values I randomly sampled, in this case, it is 100. 
One implementation I used is Conv2DTranspose. This is basically a convolutional 2D layer and a reverse pooling layer, filling more information in the surroundings. 
The tanh activation function at the end is to make sure all generated outputs are between -1 to 1, which are the bounds of color channels after shrinking from [0, 255]. 
 
To train the generator, I needed to define the GAN model. There is a zero-sum relationship between the generator and the discriminator. When the discriminator is good at detecting fake images, the weights of the generator are updated to not be flagged as fake ones. But when the discriminator is not good at detecting fake images, the generator is updated less. Therefore, the GAN model contains both models to balance their effect on each other. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159187232-594dd605-963e-408c-9355-21b5c1e7d793.png" alt="gan function" width="600px"/>
</p>

Again, there were 5,000 dog images, running in 128 batches, around 39 images in each batch.

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159187278-7d5ccf94-04bb-4f2a-82d2-f3ebf95780e6.png" alt="training function" width="600px"/>
</p>

## Dog Images
A total of 129 full epochs were trained before the program stopped. Here are some images generated from the generator after 78, 101, 123, and 128 epochs. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159187557-15f929b2-b230-402a-b237-4a19dbe4e71d.png" alt="78 epochs dog images" width="600px"/>
</p>

*78 epoch*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159187567-9c520bbf-9676-4bd3-bc4d-0f44c255995f.png" alt="101 epochs dog images" width="600px"/>
</p>

*101 epoch*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159187580-054c2c04-ed79-4c5d-be5c-ae5bce2c6bdd.png" alt="123 epochs dog images" width="600px"/>
</p>

*123 epoch*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159187595-51104cc3-3b17-46c9-9bac-7afa419d76f9.png" alt="128 epochs dog images" width="600px"/>
</p>

*128 epoch*

I observed that the images get richer in content as I increase epochs from around 78 to 123. By 78 epochs, I can spot some eyes and the contour of the face for some images, but the resolution is quite low. But after 100 epochs, I can see a variety of colors from the set of images generated. There are also more variety of eyes, ears, and faces. For this specific attempt, 123 epochs yielded strong results where I can discern some body position of the dog as well. However, such a decision is harder to achieve at 128 epochs. If possible, training more epochs could generate more compelling images that could even fool humans. 

### Generate Fake Dog Images
I saved the models for the above 4 epochs for training dog images and sample a set of random points and see how these models will represent these points. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159188274-20ec945e-0b15-4c5a-a586-ea7afb6abc29.png" alt="78 epochs dog images generated" width="600px"/>
</p>

*78 epochs*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159188278-f1dc1dee-2d69-434e-b13d-01f7289f060d.png" alt="101 epochs dog images generated" width="600px"/>
</p>

*101 epochs*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159188280-3fa509f3-a2a2-4437-b672-d2222d2740a2.png" alt="123 epochs dog images generated" width="600px"/>
</p>

*123 epochs*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159188283-9f548dfa-7be8-47b1-8754-489655bba90e.png" alt="128 epochs dog images generated" width="600px"/>
</p>

*128 epochs*

Again, the contour of the object in the image got clearer as the number of epochs increased, there is more color in the images as well. Increasing the number of epochs would likely increase the interpretability and the quality of these dog images perceived by humans. Nevertheless, these images are astonishing as they are learned only from perturbing random numbers. 

## Dog and Cat Images
I also ran another GAN model with 5,000 images of cats and dogs each as real images. The generator was trained with 113 epochs. 

Below are some images of the generator at the specified epochs. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159188897-9846e966-31ad-48e4-b6d2-0d51bd0bbad7.png" alt="15 epochs dog and cat images" width="600px"/>
</p>

*15 epochs*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159188898-2cf82ac8-d655-4746-8c68-cda2e58e4a92.png" alt="30 epochs dog and cat images" width="600px"/>
</p>

*30 epochs*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159188904-9d55fb17-2a9d-4cc9-ac84-95e3bce95d6a.png" alt="100 epochs dog and cat images" width="600px"/>
</p>

*100 epochs*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159188909-aa4aa766-890c-4397-99df-464cb8ae4d08.png" alt="109 epochs dog and cat images" width="600px"/>
</p>

*109 epochs*

<p align="center">
  <img src="https://user-images.githubusercontent.com/60123440/159188915-717065bf-9092-4cf3-81ff-528f497ae0b5.png" alt="113 epochs dog and cat images" width="600px"/>
</p>

*113 epochs*

For the epochs trained, I observed some images with some pointy ears suggesting the image of a cat, but other facial parts were not that clear for a cat face. I drew similar conclusions as above with dog images that more epochs will yield better results, including a clearer image of these animals. In addition, the output would be more favorable if the model was trained on higher resolution images. However, due to the computational power constraint, the CIFAR-10 dataset was the best choice. Nevertheless, the GAN model has high potential and encourage training GAN with high quality images and more images for more epochs to generate artificial, yet interesting and intriguing images. 
