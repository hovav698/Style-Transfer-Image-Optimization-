This repository contains a PyTorch implementation of the original Neural Style Transfer paper  ([🔗Gatys et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf))

**How the algorithm works**

The algorithm transfers style from one input image (the style image) onto another input image (the content image) using CNN net. I chose to use VGG19 for this project. 
The model outputs activations from different layers of the VGG network, each activation represent features of different scale.

The algorithm loss function consist of two losses - the content loss and the style loss:

**Style loss**: The style image and another randomly initialized image is fed into the network,  a matrix that called the gram matrix is then calculated from the activations output.
The gram matrix is simply the matrix of the inner product of each vector and its corresponding vectors in the same image. It represents correlantion between the image and itself, and this is basically the mathematical definition of style. 
For style loss minimazation, the distance between the gram matrices of all the activation of the style image to the activation of the random image should be minimal.


**Content Loss**: The content loss is an MSE loss between the original content image and the the random image. 

The two losses forces the random image to be similar to both the style image and the content image, by optimizing the random image pixel values untill the total loss is minimal. The output is a stylized image which keeps the content from the content image but takes the style from the style image. 


<img width="750" alt="STarch" src="https://user-images.githubusercontent.com/71300410/121800545-3dfc1700-cc3b-11eb-91c1-01012534fcac.png">


Style image      +       Content image     =       Style transfer image


<img src="https://user-images.githubusercontent.com/71300410/121797285-ff109600-cc27-11eb-91a9-fee190e8b734.png" width="250" height="250" />  <img src="https://user-images.githubusercontent.com/71300410/121797002-f74ff200-cc25-11eb-9e9b-b9975cace2b0.png" width="250" height="250" />
   <img src="https://user-images.githubusercontent.com/71300410/121797095-b99f9900-cc26-11eb-9e8c-2932733fdb35.png" width="250" height="250" />
   
   
   
**Content/Style tradeoff**

We can decide how much we want to preserve the original content of the image and how much style we want to add to the content image, by changing the weight of the style/content loss.

Example of choosing different style weights:

 <img src="https://user-images.githubusercontent.com/71300410/121798862-61ba5f80-cc31-11eb-9156-ceadd8bcc18f.png" width="250" height="250" /> <img src="https://user-images.githubusercontent.com/71300410/121797095-b99f9900-cc26-11eb-9e8c-2932733fdb35.png" width="250" height="250" /> <img src="https://user-images.githubusercontent.com/71300410/121798751-b8736980-cc30-11eb-806d-6405b11325c8.png" width="250" height="250" /> 



