## Scope of the project

The bmAi project aims at using Artificial Intelligence against malnutrition in developing countries.

This project was a result of a semester project, done with the supervision of the LTS5 Lab at EPFL, with the collaboration of the EssentialTech Center.


### Description.

The goal of this project is to develop a Deep Learning solution that can estimate body measures from mobile phone pictures with unconstrained settings in a rapid, easy, automatic, reproducible, and accurate manner.

We propose a solution based on Pose Detection.
Instead of detecting keypoints, we use the Part Affinity Fields detected by the OpenPose algorithm. The estimated PAFs are then fed to a seperate neural network that performs estimation of height and weight. We show that integrating AGE and SEX attributes in a seperate independent branch improved the performance significantly. 
We achieve an acceptable performance, reaching a Mean realtive error of 5.8% for height, and 9.1 for weight.
