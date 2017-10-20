![banner](https://github.com/nashory/gans-awesome-applications/blob/master/jpg/gans.jpg)

# gans-awesome-applications
Curated list of awesome GAN applications and demonstrations.  

__Note: General GAN papers targeting simple image generation such as DCGAN, BEGAN etc. are not included in the list. I mainly care about applications.__

## The landmark papers that I respect.
+ Generative Adversarial Networks, [[paper]](https://arxiv.org/abs/1406.2661), [[github]](https://github.com/goodfeli/adversarial)
+ Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1511.06434), [[github]](https://github.com/soumith/dcgan.torch)
+ Improved Techniques for Training GANs, [[paper]](https://arxiv.org/pdf/1606.03498.pdf), [[github]](https://github.com/openai/improved-gan)
+ BEGAN: Boundary Equilibrium Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1703.10717), [[github]](https://github.com/carpedm20/BEGAN-tensorflow)

-----

## Contents
__Use this contents list or simply press <kbd>command</kbd> + <kbd>F</kbd> to search for a keyword__
+ [Applications using GANs](#applications-using-gans)
    + [Font geneation](#font-generation)
    + [Anime character generation](#anime-character-generation)
    + [Interactive Image generation](#interactive-image-generation)
    + [Text2Image (text to image)](#text2image-text-to-image)
    + [3D Obejct generation](#3d-obejct-generation)
    + [Image Editing](#image-editing)
    + [Face Aging](#face-aging)
    + [Human Pose Estimation](#human-pose-estimation)
    + [Domain-transfer (e.g. style-transfer, pix2pix, sketch2image)](#domain-transfer-eg-style-transfer-pix2pix-sketch2image)
    + [Image Inpainting (hole filling)](#image-inpainting-hole-filling)
    + [Super-resolution](#super-resolution)
    + [High-resolution image generation (large-scale image)](#high-resolution-image-generation-large-scale-image)
    + [Adversarial Examples (Defense vs Attack)](#adversarial-examples-defense-vs-attack)
    + [Visual Saliency Prediction (attention prediction)](#visual-saliency-prediction-attention-prediction)
    + [Object Detection/Recognition](#object-detectionrecognition)
+ [Did not use GAN, but still interesting applications](#did-not-use-gan-but-still-interesting-applications)

    + [Real-time face reconstruction](#real-time-face-reconstruction)
    + [Super-resolution](#super-resolution-1)
    + [Photorealistic Image generation (e.g. pix2pix, sketch2image)](#photorealistic-image-generation-eg-pix2pix-sketch2image)
    + [Human Pose Estimation](#human-pose-estimation-1)
    + [3D Object generation](#3d-object-generation-1)
+ [GAN tutorials with easy and simple example code for starters](#gan-tutorials-with-easy-and-simple-example-code-for-starters)
+ [Implementations of various types of GANs collection](#implementations-of-various-types-of-gans-collection)


-----

## Applications using GANs

### Font generation
+ Learning Chinese Character style with conditional GAN, [[blog]](https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html), [[github]](https://github.com/kaonashi-tyc/zi2zi)

### Anime character generation
+ Towards the Automatic Anime Characters Creation with Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1708.05509)
+ [Project] A simple PyTorch Implementation of Generative Adversarial Networks, focusing on anime face drawing, [[github]](https://github.com/jayleicn/animeGAN)
+ [Project] A simple, clean TensorFlow implementation of Generative Adversarial Networks with a focus on modeling illustrations, [[github]](https://github.com/tdrussell/IllustrationGAN)
+ [Project] Keras-GAN-Animeface-Character, [[github]](https://github.com/forcecore/Keras-GAN-Animeface-Character)
+ [Project] A DCGAN to generate anime faces using custom mined dataset, [[github]](https://github.com/pavitrakumar78/Anime-Face-GAN-Keras)

### Interactive Image generation
+ Generative Visual Manipulation on the Natural Image Manifold, [[paper]](https://arxiv.org/pdf/1609.03552), [[github]](https://github.com/junyanz/iGAN)
+ Neural Photo Editing with Introspective Adversarial Networks, [[paper]](http://arxiv.org/abs/1609.07093), [[github]](https://github.com/ajbrock/Neural-Photo-Editor)

### Text2Image (text to image)
+ TAC-GAN – Text Conditioned Auxiliary Classifier Generative Adversarial Network, [[paper]](https://arxiv.org/pdf/1703.06412.pdf), [[github]](https://github.com/dashayushman/TAC-GAN)
+ StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1612.03242.pdf), [[github]](https://github.com/hanzhanggit/StackGAN)
+ Generative Adversarial Text to Image Synthesis, [[paper]](https://arxiv.org/pdf/1605.05396.pdf), [[github]](https://github.com/paarthneekhara/text-to-image), [[github]](https://github.com/reedscot/icml2016)
+ Learning What and Where to Draw, [[paper]](http://www.scottreed.info/files/nips2016.pdf), [[github]](https://github.com/reedscot/nips2016)

### 3D Object generation
+ Parametric 3D Exploration with Stacked Adversarial Networks, [[github]](https://github.com/maxorange/pix2vox), [[youtube]](https://www.youtube.com/watch?v=ITATOXVvWEM)
+ Learning a Probabilistic Latent Space of Object
Shapes via 3D Generative-Adversarial Modeling, [[paper]](http://papers.nips.cc/paper/6096-learning-a-probabilistic-latent-space-of-object-shapes-via-3d-generative-adversarial-modeling.pdf), [[github]](https://github.com/zck119/3dgan-release), [[youtube]](https://www.youtube.com/watch?v=HO1LYJb818Q)
+ 3D Shape Induction from 2D Views of Multiple Objects, [[paper]](https://arxiv.org/pdf/1612.05872.pdf)
+ Fully Convolutional Refined Auto-Encoding Generative Adversarial Networks for 3D Multi Object Scenes, [[github]](https://github.com/yunishi3/3D-FCR-alphaGAN), [[blog]](https://becominghuman.ai/3d-multi-object-gan-7b7cee4abf80)

### Image Editing
+ Invertible Conditional GANs for image editing, [[paper]](https://arxiv.org/abs/1611.06355), [[github]](https://github.com/Guim3/IcGAN)
+ Image De-raining Using a Conditional Generative Adversarial Network, [[paper]](https://arxiv.org/abs/1701.05957), [[github]](https://github.com/hezhangsprinter/ID-CGAN)

### Face Aging
+ Age Progression/Regression by Conditional Adversarial Autoencoder, [[paper]](https://arxiv.org/pdf/1702.08423), [[github]](https://github.com/ZZUTK/Face-Aging-CAAE)
+ CAN: Creative Adversarial Networks Generating “Art” by Learning About Styles and Deviating from Style Norms, [[paper]](https://arxiv.org/pdf/1706.07068.pdf)
+ FACE AGING WITH CONDITIONAL GENERATIVE ADVERSARIAL NETWORKS, [[paper]](https://arxiv.org/pdf/1702.01983.pdf)

### Human Pose Estimation
+ Pose Guided Person Image Generation, [[paper]](https://arxiv.org/abs/1705.09368)

### Domain-transfer (e.g. style-transfer, pix2pix, sketch2image)
+ Image-to-Image Translation with Conditional Adversarial Networks, [[paper]](https://arxiv.org/pdf/1611.07004), [[github]](https://github.com/phillipi/pix2pix), [[youtube]](https://www.youtube.com/watch?v=VVqxbmUJorQ)
+ Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, [[paper]](https://arxiv.org/pdf/1703.10593.pdf), [[github]](https://github.com/junyanz/CycleGAN), [[youtube]](https://www.youtube.com/watch?v=JzgOfISLNjk)
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

### High-resolution image generation (large-scale image)
+ Generating Large Images from Latent Vectors, [[blog]](http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/), [[github]](https://github.com/hardmaru/cppn-gan-vae-tensorflow)

### Adversarial Examples (Defense vs Attack) 
+ SafetyNet: Detecting and Rejecting Adversarial Examples Robustly, [[paper]](https://arxiv.org/abs/1704.00103)
+ ADVERSARIAL EXAMPLES FOR GENERATIVE MODELS, [[paper]](https://arxiv.org/pdf/1702.06832.pdf), [[github]]()
+ Adversarial Examples Generation and Defense Based on Generative Adversarial Network, [[paper]](http://cs229.stanford.edu/proj2016/report/LiuXia-AdversarialExamplesGenerationAndDefenseBasedOnGenerativeAdversarialNetwork-report.pdf)


### Visual Saliency Prediction (attention prediction)
+ SalGAN: Visual Saliency Prediction with Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1701.01081), [[github]](https://github.com/imatge-upc/saliency-salgan-2017)

### Object Detection/Recognition
+ Perceptual Generative Adversarial Networks for Small Object Detection, [[paper]](https://arxiv.org/pdf/1706.05274)
+ Adversarial Generation of Training Examples for Vehicle License Plate Recognition, [[paper]](https://arxiv.org/pdf/1707.03124.pdf)

### Robotics
+ Unsupervised Pixel–Level Domain Adaptation with Generative Adversarial Networks, [[paper]](https://arxiv.org/pdf/1612.05424.pdf), [[github]](https://github.com/rhythm92/Unsupervised-Pixel-Level-Domain-Adaptation-with-GAN)

### Video (generation/prediction)
+ DEEP MULTI-SCALE VIDEO PREDICTION BEYOND MEAN SQUARE ERROR, [[paper]](https://arxiv.org/pdf/1511.05440.pdf), [[github]](https://github.com/dyelax/Adversarial_Video_Generation)

### Synthetic Data Generation
+ Learning from Simulated and Unsupervised Images through Adversarial Training, [[paper]](https://arxiv.org/pdf/1612.07828.pdf), [[github]](https://github.com/carpedm20/simulated-unsupervised-tensorflow)

### Others
+ (Physics) Learning Particle Physics by Example:
Location-Aware Generative Adversarial Networks for
Physics Synthesis, [[paper]](https://arxiv.org/pdf/1701.05927.pdf), [[github]](https://github.com/hep-lbdl/adversarial-jets)


-----

## Did not use GAN, but still interesting applications.

### Real-time face reconstruction
+ Model-based Deep Convolutional Face Autoencoder for Unsupervised Monocular Reconstruction, [[paper]](https://arxiv.org/pdf/1703.10580.pdf), [[github]](https://github.com/waxz/MoFA), [[youtube]](https://www.youtube.com/watch?v=uIMpHZYB8fI)

### Super-resolution
+ Learning to Simplify:
Fully Convolutional Networks for Rough Sketch Cleanup, [[paper]](http://delivery.acm.org/10.1145/2930000/2925972/a121-simo-serra.pdf?ip=111.91.137.238&id=2925972&acc=ACTIVE%20SERVICE&key=58C7DD92F91E3631%2E58C7DD92F91E3631%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&CFID=818332500&CFTOKEN=94661101&__acm__=1507786813_0e5b28dfb97e654d0126d61b0aa592f4), [[site link]](http://hi.cs.waseda.ac.jp/~esimo/en/research/sketch/), [[youtube]](https://www.youtube.com/watch?v=4MfG9CDufPA)

### Photorealistic Image generation (e.g. pix2pix, sketch2image)
+ The Sketchy Database: Learning to Retrieve Badly Drawn Bunnies, [[paper]](http://delivery.acm.org/10.1145/2930000/2925954/a119-sangkloy.pdf?ip=111.91.137.238&id=2925954&acc=CHORUS&key=58C7DD92F91E3631%2E58C7DD92F91E3631%2E4D4702B0C3E38B35%2E6D218144511F3437&CFID=818332500&CFTOKEN=94661101&__acm__=1507787415_cb950c300370fc27da68920a0d5b5178), [[youtube]](https://www.youtube.com/watch?v=a3sgFQjEfp4)
+ PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing, [[paper]](https://www.researchgate.net/profile/Eli_Shechtman/publication/220184392_PatchMatch_A_Randomized_Correspondence_Algorithm_for_Structural_Image_Editing/links/02e7e520897b12bf0f000000.pdf), [[github]](https://github.com/younesse-cv/PatchMatch), [[youtube]](https://www.youtube.com/watch?v=n3aoc36V8LM)

### Human Pose Estimation
+ Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation, [[paper]](https://arxiv.org/pdf/1705.02407.pdf), [[github]](https://github.com/Guanghan/GNet-pose)

### 3D Obejct generation
+ 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction, [[paper]](http://arxiv.org/abs/1604.00449), [[github]](https://github.com/chrischoy/3D-R2N2)

-----

## GAN tutorials with easy and simple example code for starters
+ [1D Generative Adversarial Network Demo](http://notebooks.aylien.com/research/gan/gan_simple.html)
+ [starter from "How to Train a GAN?" at NIPS2016](https://github.com/soumith/ganhacks)
+ [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)
+ [OpenAI - Generative Models](https://blog.openai.com/generative-models/)
+ [[paper]](), [[github]](), [[youtube]]()

----

## Implementations of various types of GANs collection
+ [nashory/gans-collections.torch](https://github.com/nashory/gans-collection.torch), torch7
+ [hwalsuklee/tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections), pytorch
+ [wiseodd/generative-models](https://github.com/wiseodd/generative-models), both pytorch and tensorflow


___

## Trendy AI-application Articles
+ [Artificial intelligence can say yes to the dress](https://qz.com/1090267/artificial-intelligence-can-now-show-you-how-those-pants-will-fit/)




## Author
Minchul Shin, [@nashory](https://github.com/nashory)  

__Any recommendations to add to the list are welcome :)__  
__Feel free to make pull requests!__
