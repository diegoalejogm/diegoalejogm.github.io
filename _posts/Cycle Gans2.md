# Cyclic Generative Networks: CycleGans

### “ there are others who with the help of their intelligence, transform a yellow spot into sun” -- Pablo Picasso

*This article aims to explain the inner workings of the Cyclic Gans and display their capacity to change the digital world.*

<img src="/Users/diego/Downloads/horse2zebra.gif" alt="horse2zebra" style="zoom:150%;" />

A Cyclic Generative network transforming raw footage of a Horse into a Zebra.

#### Introduction

Cyclic Generative Adversarial networks (**CycleGans**) [1] are powerful computer algorithms with proven potential to revolutionize the digital ecosystem. They are capable of converting information from one representation to another one. For example, they can blur images,  color black & white images and improve image sharpness, filling gaps in pictures and converting the genre of a music piece.

They are more powerful than your conventional design/production/writing platforms. Because **CycleGans** are machine learning algorithms, they can in principle learn whatever transformation is desired. Contrarily, traditional transformation software is hardcoded and is developed to execute a single specific task. In addition, **CycleGans** achieve higher performance than existing software, since they can learn from data and improve as more of it is collected.

Understanding their workings and capacities at different levels is exciting and provides insights on how Artificial Intelligence can impact our day to day in unprecedented ways. 

https://www.youtube.com/watch?v=ZYSY8Ypzbus&list=PLJ_xKSPzfY_BsCJlkBFMLLQqRM9RIxQuM&index=3

CycleGan transforming 2D colored representations of buildings into hand-made sketches with high details.

#### GANs

Before talking about CycleGans, let's briefly discuss regular Generative Adversarial Networks (GANs).

Generative adversarial networks [2] are [machine learning algorithms](https://en.wikipedia.org/wiki/Machine_learning) capable of creating data. When they are feed information such as images, sounds or text, they learn to generate new similarly looking/sounding outputs.  For example: given a set of human face images, the computer program can *teach itself* (*train*, in the Machine Learning jargon) what a human face looks like, and be able to create accurate faces. I encourage to take a look at [this article](https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f), which explains the fundamentals of Gans in an understandable way.

![stylegan2-teaser-1024x256.png](https://github.com/NVlabs/stylegan2/blob/master/docs/stylegan2-teaser-1024x256.png?raw=true)

Artificially generated faces using GANs, by NVIDIA research. Taken from [4].

Cyclic GANs are a particular variation over traditional GANs. They can also create new samples of data, but they do this by transforming input samples, rather than creating ouptuts from scratch. In other words, they learn to transform data from a two distributions of data;  these distributions can be chosen by the scientist or developer who provides the sets of data to this algorithm. Assuming that the elements of `C` are pictures of **C**ats, and the elements of `D` are pictures of **D**ogs, the algorithm is effectively capable of converting a cat image to a dog image, and viceversa. 

## Inner workings

#### Bijections

Effectively, a CycleGan is learning a data transformation function between two domains. Such a transformation `F(c)`, converts elements from domain *`C`* into elements of domain *`D`*. Simultaneously, the CycleGan learns a function `G(d)` that transforms elements from *`D`* into elements of domain *`C`*. This can be re-written as:

<img src="/Users/diego/Downloads/Bijection.png" alt="Bijection" style="zoom:33%;" />

Both F and G are bijections [4] which means that for a given value `c1 ⋲ C ` there is a single representation `d1 ⋲ D`.

<img src="/Users/diego/Downloads/Gen_bijection.svg.png" alt="Gen_bijection.svg" style="zoom: 25%;" />

#### GAN loss

Two traditional GANs are used to learn the two desired and opposite transformations `C` and `G`. Each GAN is individually trained to learn one of these functions, by minimizing an error or loss that depends on its predictions. If the outputs of a GAN are too diferent to the targets, they will be highly penalized. Our Cyclic network will then incorporate two losses, one for each GAN.

![gans-loss](/Users/diego/Documents/github/diegoalejogm.github.io/_posts/img/gans-loss.png)

A GAN constists of two networks that "play" an adversarial game. Given a dataset containing the target samples to be recreated and fake ones, a **Discriminator** network will learn to distinguish the real from the fake, while a **Generator** network will learn to trick the Discriminator and reproduce similar data to the desired one. The GAN loss aims to maximize the reproduction of data similar to the target one.

![gans-schema](/Users/diego/Documents/github/diegoalejogm.github.io/_posts/img/gans-schema.png)

Representation of a GAN

#### Cycle loss

CycleGans apply the concept of **transformation consistency**. The GAN error function does not guarantee that transforming an image back and forth would result in the same input you originally had. A more meaninful representation would be cyclic/consistent, as it would enable the network to learn a better approximation of the data distribution (e.g. black and white images). Transformation consistency links the two separate GANs `F` and `G`, and reduce the possible set of mappings that these networks can learn, those that provide more meaningful representations.

A cycle consistency loss is used to enforce transformation consistency when training CycleGans. It is applied in addition to the former GAN loss described above, summing it as a [regularization term](https://en.wikipedia.org/wiki/Regularization_(mathematics)) . Given loss is defined as the L1 norm of the forward-cycle predictions (see image below). If `F` and `G` are the learned transformations, this loss represents the absolute difference between an input value `x` and it's forward-cycle prediction `F(G(x))`. The higher the absolute difference between these values, the more distant the predictions are from the original inputs. Note that an additional L1 norm is needed using input values `y` and their forward-cycle prediciton `G(F(y))`.

![cyclegan-cycle-loss](/Users/diego/Documents/github/diegoalejogm.github.io/_posts/img/cyclegan-cycle-loss.png)

#### Total loss

The final CycleGan loss that is used to train the network is:

![cyclegan-total-loss](/Users/diego/Documents/github/diegoalejogm.github.io/_posts/img/cyclegan-total-loss.png)

### Results

The concepts above have been applied to several tasks in machine learning. They prove the feasability and potential of applying CycleGans to diverse domains. Some of them are exemplified next:

##### Photo enhancement

![img](https://junyanz.github.io/CycleGAN/images/photo_enhancement.jpg)

A CycleGan was trained to generate professionally looking pictures of flowers, with several levels of focus and blurriness.

##### Image Style transfer

<img src="https://junyanz.github.io/CycleGAN/images/photo2painting.jpg" alt="img" style="zoom: 50%;" />

Consists of transforming the style of source images to a different one. For example, it is possible to transform photographies into painting-like images, similar to what a given artist like Van Gogh would have painted it like.

#### Music genre transfer 

Researchers from ETH univeristy in Zurich were able to train a CycleGAN to convert classical music pieces to jazz, and viceversa [4].

https://www.youtube.com/watch?v=wbxjsqUFsXg

#### Voice conversion

Researchers from NTT Communication Science Laboratories in Japan demonstrated astonishing results when using CycleGANs to convert voice between locutors, regardless of their gender [5].

http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html

#### CycleGan Issues

Despite the apparent success of CycleGANs in several tasks, they have yet to be 100% accurate. Here are some of their current pitfalls.

<img src="https://camo.githubusercontent.com/757b691307b52fe8a0806dde3a560dc068dbf5b3/68747470733a2f2f6a756e79616e7a2e6769746875622e696f2f4379636c6547414e2f696d616765732f6661696c7572655f707574696e2e6a7067" alt="img" style="zoom:50%;" />

1. CycleGans may provide unexpected results when fed input data that differs the one in which it was trained. The CycleGan used to transform horse and zebra images was not fed inputs of human beings, thus it could generate arbitrary transformations.
   
2. Tasks that require geometric changes, instead of color or contrast, result in minimal changes to the input. 

#### Final words

Congratulations to [Jun-Yan Zhu](http://www.eecs.berkeley.edu/~junyanz/)*  [Taesung Park](https://taesung.me/)*  [Phillip Isola](http://web.mit.edu/phillipi/)  [Alexei A. Efros](http://www.eecs.berkeley.edu/~efros/) from the [AI Research lab at UC Berkeley](http://bair.berkeley.edu/) for their work inventing CycleGans and showing their potential. You can find their website, which includes more information related to the project [https://junyanz.github.io/CycleGAN/](https://junyanz.github.io/CycleGAN/).

Feel free to take a look at [my GANs repository](http://github.com/diegoalejogm/gans), where you will find my ongoing CycleGan implementation from scratch in PyTorch and Tensorflow. A tutorial through Medium should be found once I have finished it.

#### References

[1] [Jun-Yan Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J), [Taesung Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+T), [Phillip Isola](https://arxiv.org/search/cs?searchtype=author&query=Isola%2C+P), [Alexei A. Efros](https://arxiv.org/search/cs?searchtype=author&query=Efros%2C+A+A), *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*, https://arxiv.org/abs/1703.10593

[2] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio, *Generative Adversarial Networks*, 2014, https://arxiv.org/abs/1406.2661

[3] [Tero Karras](https://arxiv.org/search/cs?searchtype=author&query=Karras%2C+T), [Samuli Laine](https://arxiv.org/search/cs?searchtype=author&query=Laine%2C+S), [Miika Aittala](https://arxiv.org/search/cs?searchtype=author&query=Aittala%2C+M), [Janne Hellsten](https://arxiv.org/search/cs?searchtype=author&query=Hellsten%2C+J), [Jaakko Lehtinen](https://arxiv.org/search/cs?searchtype=author&query=Lehtinen%2C+J), [Timo Aila](https://arxiv.org/search/cs?searchtype=author&query=Aila%2C+T), *Analyzing and Improving the Image Quality of StyleGAN*, https://arxiv.org/abs/1912.04958

[4] [Gino Brunner](https://arxiv.org/search/cs?searchtype=author&query=Brunner%2C+G), [Yuyi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Roger Wattenhofer](https://arxiv.org/search/cs?searchtype=author&query=Wattenhofer%2C+R), [Sumu Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+S), Symbolic Music Genre Transfer with CycleGAN,  https://arxiv.org/abs/1809.07575)

[5] [Takuhiro Kaneko](https://arxiv.org/search/cs?searchtype=author&query=Kaneko%2C+T), [Hirokazu Kameoka](https://arxiv.org/search/cs?searchtype=author&query=Kameoka%2C+H), [Kou Tanaka](https://arxiv.org/search/cs?searchtype=author&query=Tanaka%2C+K), [Nobukatsu Hojo](https://arxiv.org/search/cs?searchtype=author&query=Hojo%2C+N). CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion, https://arxiv.org/abs/1904.04631