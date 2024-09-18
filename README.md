# HI-192

Group project for HI 192 (Knowledge Representation & Health Decision Support System) 

- Glaucoma Detection using Convolutional Neural Networks
- Dataset used includes 307 retinal images produced using the OpenAI API
- 152 images diagnosed with glaucoma and 155 healthy eye images
- Utilizes Python as the main programming language
- Utilizes keras and tensorflow
- Dataset underwent preprocessing such as normalization, image shuffling, and augmentation before an 80-20 train-test split
- Training set underwent another split (70-30 train-validation split)
- Compared the predictive performance of 3 base CNN models to 3 existing models (VGG19, ResNet50, InceptionV3)
- Best performing models are the VGG19 and InceptionV3 models
