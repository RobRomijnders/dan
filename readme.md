# A Domain Agnostic Normalization Layer for Unsupervised Adversarial Domain Adaptation
R. Romijnders, P.Meletis, G. Dubbelman
Eindhoven, University of technology
TU/e-SPS (VCA): Mobile Perception Systems
Delivery date: July 3rd 2018

Upon acceptance, we will publish the code in this repository to reproduce all our experiments and results.

Note of July 17: this code will be made public after first review round of IEEE WACV (something like October 2018)

Notes for instruction:


  * Best starting point is to change `scripts/set_env_names.sh`
    * Then run a script like `train_many.sh` or `evaluate_many.sh`
  * The scripts in `train.py` and `evaluate.py` can be run from command line

Notes on versions

  * `doc/requirements.txt` contains the output of `pip freeze` run on July 17, 2018
  * This code base started as fork from the `semantic-segmentation/v0.7` code by Panos Meletis. Last fork on last week October, 2017
  * 


Repository structure

  * Estimator: all code necessary to run via tf.estimator API
  * Input: all code related to the input pipeline of the data. We use the tf.data API
  * Misc: Miscellaneous code. Mainly contains code for plotting
  * Model: the actual model for the __representation learner__, __segmenter__ and __domain classifier__
  * Scripts: all code for shell scripting. Note, I learned bash along this project, so this code probably has some beginner-mistakes
  * Utils: all kinds of utility functions
