#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/test_classifier.py
#
# PROGRAMMER: Jennifer S.
# DATE CREATED: 01/30/2018
# REVISED DATE:             <=(Date Revised - if any)
# PURPOSE: To demonstrate the proper usage of the classifier() function that
#          is defined in classifier.py This function uses CNN model
#          architecture that has been pretrained on the ImageNet data to
#          classify images. The only model architectures that this function
#          will accept are: 'resnet', 'alexnet', and 'vgg'. See the example
#          usage below.
#
# Usage: python test_classifier.py    -- will run program from commandline

# Imports classifier function for using pretrained CNN to classify images
from classifier import classifier
import time

# Defines a dog test image from pet_images folder
test_image="pet_images/Collie_03797.jpg"

# Defines a model architecture to be used for classification
# NOTE: this function only works for model architectures:
#      'vgg', 'alexnet', 'resnet'
model = "alexnet"

# Demonstrates classifier() functions usage
# NOTE: image_classication is a text string - It contains mixed case(both lower
# and upper case letter) image labels that can be separated by commas when a
# label has more than one word that can describe it.

start_time = time.time()
image_classification = classifier(test_image, model)
end_time = time.time()
run_time = end_time - start_time
# prints result from running classifier() function
print("\nResults from test_classifier.py\nImage:", test_image, "using model:",
      model, "was classified as a:", image_classification)
print("\nRun Time: {}".format(str(run_time)))
