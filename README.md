# gans-awesome-applications
Curated list of awesome GAN applications and demonstrations.  

__Note: General GAN papers targeting simple image generation such as DCGAN, BEGAN etc. are not included in the list. I mainly care about applications.__

## The landmark papers that I respect.
+ Generative Adversarial Networks, [[paper]](https://arxiv.org/abs/1406.2661), [[github]](https://github.com/goodfeli/adversarial)
+ Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1511.06434), [[github]](https://github.com/soumith/dcgan.torch)
+ BEGAN: Boundary Equilibrium Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1703.10717), [[github]](https://github.com/carpedm20/BEGAN-tensorflow)

-----

## Applications using GANs

### Font generation
+ Learning Chinese Character style with conditional GAN, [[blog]](https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html), [[github]](https://github.com/kaonashi-tyc/zi2zi)

### Anime character generation
+ Towards the Automatic Anime Characters Creation with Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1708.05509)
+ A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing, [[github]](https://github.com/jayleicn/animeGAN)

### Interactive Image generation
+ Generative Visual Manipulation on the Natural Image Manifold, [[paper]](https://arxiv.org/pdf/1609.03552), [[github]](https://github.com/junyanz/iGAN)
+ Neural Photo Editing with Introspective Adversarial Networks, [[paper]](http://arxiv.org/abs/1609.07093), [[github]](https://github.com/ajbrock/Neural-Photo-Editor)

### Text2Image (text to image)
+ TAC-GAN – Text Conditioned Auxiliary Classifier Generative Adversarial Network, [[paper]](https://arxiv.org/pdf/1703.06412.pdf), [[github]](https://github.com/dashayushman/TAC-GAN)
+ StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1612.03242.pdf), [[github]](https://github.com/hanzhanggit/StackGAN)
+ Generative Adversarial Text to Image Synthesis, [[paper]](https://arxiv.org/pdf/1605.05396.pdf), [[github]](https://github.com/paarthneekhara/text-to-image), [[github]](https://github.com/reedscot/icml2016)
+ Learning What and Where to Draw, [[paper]](http://www.scottreed.info/files/nips2016.pdf), [[github]](https://github.com/reedscot/nips2016)

### 3D Obejct generation
+ Parametric 3D Exploration with Stacked Adversarial Networks, [[github]](https://github.com/maxorange/pix2vox), [[Youtube]](https://www.youtube.com/watch?v=ITATOXVvWEM)
+ Learning a Probabilistic Latent Space of Object
Shapes via 3D Generative-Adversarial Modeling, [[paper]](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf), [[github]](https://github.com/zck119/3dgan-release), [[Youtube]](https://www.youtube.com/watch?v=HO1LYJb818Q)

### Image Editing
+ Invertible Conditional GANs for image editing [[paper]](https://arxiv.org/abs/1611.06355), [[github]](https://github.com/Guim3/IcGAN)

### Face Aging
+ Age Progression/Regression by Conditional Adversarial Autoencoder, [[paper]](https://arxiv.org/pdf/1702.08423), [[github]](https://github.com/ZZUTK/Face-Aging-CAAE)
+ CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms, [[paper]](https://arxiv.org/pdf/1706.07068.pdf)
+ [FACE AGING WITH CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1702.01983.pdf), [[github]]()

### Human Pose Estimation
+ Pose Guided Person Image Generation, [[paper]](https://arxiv.org/abs/1705.09368)

### Domain-transfer (e.g. style-transfer, pix2pix, sketch2image)
+ Image-to-Image Translation with Conditional Adversarial Networks, [[paper]](https://arxiv.org/pdf/1611.07004), [[github]](https://github.com/phillipi/pix2pix), [[Youtube]](https://www.youtube.com/watch?v=VVqxbmUJorQ)
+ Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, [[paper]](https://arxiv.org/pdf/1703.10593.pdf), [[github]](https://github.com/junyanz/CycleGAN), [[Youtube]](https://www.youtube.com/watch?v=JzgOfISLNjk)
+ Learning to Discover Cross-Domain Relations with Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1703.05192.pdf), [[github]](https://github.com/carpedm20/DiscoGAN-pytorch)
+ Unsupervised Creation of Parameterized Avatars, [[paper]](https://arxiv.org/pdf/1704.05693.pdf)
+ UNSUPERVISED CROSS-DOMAIN IMAGE GENERATION, [[paper]](https://openreview.net/pdf?id=Sk2Im59ex)
+ Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks, [[paper]](http://arxiv.org/abs/1604.04382), [[github]](https://github.com/chuanli11/MGANs)
+ Pixel-Level Domain Transfer  [[paper]](https://arxiv.org/pdf/1603.07442), [[github]](https://github.com/fxia22/PixelDTGAN)
+ TextureGAN: Controlling Deep Image Synthesis with Texture Patches, [[paper]](https://arxiv.org/pdf/1706.02823.pdf), [[demo]](https://github.com/varunagrawal/t-gan-demo)


### Image Inpainting (hole filling)
+ Context Encoders: Feature Learning by Inpainting, [[paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pathak_Context_Encoders_Feature_CVPR_2016_paper.pdf), [[github]](https://github.com/pathak22/context-encoder)
+ Semantic Image Inpainting with Perceptual and Contextual Losses, [[paper]](https://arxiv.org/abs/1607.07539), [[github]](https://github.com/bamos/dcgan-completion.tensorflow)
+ SEMI-SUPERVISED LEARNING WITH CONTEXT-CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS, [[paper]](https://arxiv.org/pdf/1611.06430v1.pdf)
+ Generative Face Completion, [[paper]](https://drive.google.com/file/d/0B8_MZ8a8aoSeenVrYkpCdnFRVms/edit), [[github]](https://github.com/Yijunmaverick/GenerativeFaceCompletion)

### Super-resolution
+ Image super-resolution through deep learning, [[github]](https://github.com/david-gpu/srez)
+ Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, [[paper]](https://arxiv.org/abs/1609.04802), [[github]](https://github.com/leehomyc/Photo-Realistic-Super-Resoluton)
+ High-Quality Face Image Super-Resolution Using Conditional Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1707.00737.pdf)

### Image Blending
+ GP-GAN: Towards Realistic High-Resolution Image Blending, [[paper]](https://arxiv.org/abs/1703.07195), [[github]](https://github.com/wuhuikai/GP-GAN)

### Generating High-resolution image (large-scale image)
+ Generating Large Images from Latent Vectors, [[blog]](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/), [[github]](https://github.com/hardmaru/cppn-gan-vae-tensorflow)



### Visual Saliency Prediction (attention prediction)
+ SalGAN: Visual Saliency Prediction with Generative Adversarial Networks [[paper]](https://arxiv.org/pdf/1701.01081), [[github]](https://github.com/imatge-upc/saliency-salgan-2017)

-----

## Did not use GAN, but still interesting applications.

### Real-time face reconstruction
+ Model-based Deep Convolutional Face Autoencoder for Unsupervised Monocular Reconstruction, [[paper]](https://arxiv.org/pdf/1703.10580.pdf), [[github]](), [[Youtube]](https://www.youtube.com/watch?v=uIMpHZYB8fI)

### Super-resolution
+ Learning to Simplify:
Fully Convolutional Networks for Rough Sketch Cleanup, [[paper]](http://delivery.acm.org/10.1145/2930000/2925972/a121-simo-serra.pdf?ip=111.91.137.238&id=2925972&acc=ACTIVE%20SERVICE&key=58C7DD92F91E3631%2E58C7DD92F91E3631%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=818332500&CFTOKEN=94661101&__acm__=1507786813_0e5b28dfb97e654d0126d61b0aa592f4), [[site link]](http://hi.cs.waseda.ac.jp/~esimo/en/research/sketch/), [[Youtube]](https://www.youtube.com/watch?v=4MfG9CDufPA)

### Photorealistic Image geneation (e.g. pix2pix, sketch2image)
+ The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies, [[paper]](http://delivery.acm.org/10.1145/2930000/2925954/a119-sangkloy.pdf?ip=111.91.137.238&id=2925954&acc=CHORUS&key=58C7DD92F91E3631%2E58C7DD92F91E3631%2E4D4702B0C3E38B35%2E6D218144511F3437&CFID=818332500&CFTOKEN=94661101&__acm__=1507787415_cb950c300370fc27da68920a0d5b5178), [[Youtube]](https://www.youtube.com/watch?v=a3sgFQjEfp4)
+ PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing, [[paper]](https://www.researchgate.net/profile/Eli_Shechtman/publication/220184392_PatchMatch_A_Randomized_Correspondence_Algorithm_for_Structural_Image_Editing/links/02e7e520897b12bf0f000000.pdf), [[github]](https://github.com/younesse-cv/PatchMatch), [[Youtube]](https://www.youtube.com/watch?v=n3aoc36V8LM)

### Human Pose Estimation
+ Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation, [[paper]](https://arxiv.org/pdf/1705.02407.pdf), [[github]](https://github.com/Guanghan/GNet-pose)

-----

## GAN tutorials with easy and simple example code for starters.
+ [1D Generative Adversarial Network Demo](http://notebooks.aylien.com/research/gan/gan_simple.html)
+ [starter from "How to Train a GAN?" at NIPS2016](https://github.com/soumith/ganhacks)
+ [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)
+ [OpenAI - Generative Models](https://blog.openai.com/generative-models/)
+ [[paper]](), [[github]](), [[Youtube]]()

----

## Implementations of various types of GANs collection
+ [nashory/gans-collections.torch](https://github.com/nashory/gans-collection.torch), torch7
+ [hwalsuklee/tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections), pytorch
+ [wiseodd/generative-models](https://github.com/wiseodd/generative-models), both pytorch and tensorflow


## Author
Minchul Shin, [@nashory](https://github.com/nashory)  

__Any recommendations to add to the list are welcome :)__  
__Feel free to make pull requests!__
