
<div align="center">
<p> Fast Deformable Linear Objects Instance Segmentation and Modelling </p>
</div>

# Abstract
In this paper, an approach for fast and accurate segmentation of Deformable Linear Objects (DLOs) named FASTDLO is presented.  
A deep convolutional neural network is employed for background segmentation, generating a binary mask that isolates DLOs in the image. Thereafter, the obtained mask is processed with a skeletonization algorithm and the intersections between different DLOs are solved with a Similarity-based network. In addition to the usual pixel-wise color-mapped image, FASTDLO also provides a spline model in 2D coordinates for each detected DLO. Synthetically generated data are exploited for the training of the data-driven methods, avoiding expensive annotations of real data. FASTDLO is experimentally compared against both a DLO-specific segmentation approach and general-purpose deep learning instance segmentation models, achieving better overall performances and a processing rate higher than 20 FPS. 

*Currently under review at Robotics and Automation Letters (RAL)*


# Installation

Main dependencies:

```
python (3.8)
pytorch (1.7.1)
opencv (4.5.1)
pillow (8.4)
scikit-image (0.18.3)
scipy (1.6.2)
shapely (1.7.1)
```

Installation (from inside the main project directory):
```
pip install .
```

# Models' weights

Download the [weights](https://drive.google.com/file/d/1_50g28B78R01ZW4_v4Gc6baQyiG-1pVE/view?usp=sharing) and place them inside a ```weights``` folder.


# Usage

import as a standard python package with ```from fastdlo.core import Pipeline```.

Then initialize the class ``` p = Pipeline(checkpoint_siamese_network, checkpoint_segmentation_network) ```

the inference can be obtained with ```pred = p.run(source_img) ```.


### Acknowledgements/Fundings
This work was supported by the European Commission’s Horizon 2020 Framework
Programme with the project REMODEL - Robotic technologies for the manipulation of complex deformable linear objects - under grant agreement No 870133.

  
DeepLabV3+ implementation based on [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)


