# Real-time Domain Adaptation in Semantic Segmentation

## Authors
- Tommaso Mazzarini (Politecnico di Torino) - tommaso.mazzarini@studenti.polito.it
- Leonardo Merelli (Politecnico di Torino) - leonardo.merelli@studenti.polito.it
- Giovanni Stinà (Politecnico di Torino) - giovanni.stina@studenti.polito.it

## Abstract
This work focuses on domain adaptation techniques to enhance the accuracy of real-time neural networks for semantic segmentation, particularly in transitioning from synthetic to real-world environments. Classical (DeepLabV2) and real-time (BiSeNet) segmentation networks are trained and evaluated on the Cityscapes dataset. Techniques such as data augmentation and adversarial learning are applied to address the domain shift problem.

## Introduction
Semantic segmentation is critical in computer vision, requiring accurate pixel-wise classification for applications like autonomous driving. Real-time networks must balance accuracy and efficiency, often challenged by the scarcity of labeled real-world data. This study investigates domain adaptation to improve model performance when shifting from synthetic to real-world data.

## Methods
- **DeepLabV2**: A classical semantic segmentation network.
- **BiSeNet**: A real-time segmentation network.
- **Domain Adaptation Techniques**: Include data augmentation and adversarial learning to mitigate the domain shift from synthetic (GTA5) to real-world (Cityscapes) datasets.

## Results
- **Baseline Performance**: Without adaptation, BiSeNet shows significant performance drops when trained on synthetic data and tested on real-world data.
- **Data Augmentation**: Improved mIoU from 20.61% to 27.43%.
- **Adversarial Learning**: Further increased mIoU to 30.44%.

## Conclusion
Domain adaptation techniques significantly improve the performance of real-time semantic segmentation models in real-world applications. While notable advancements were achieved, further research is needed to bridge the performance gap entirely.
