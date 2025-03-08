# 
# Landcover classification of Catalonia with Satellite images

Final Project for the UPC [Artificial Intelligence with Deep Learning Postgraduate Course](https://www.talent.upc.edu/ing/estudis/formacio/curs/310402/postgraduate-course-artificial-intelligence-deep-learning/) 2024-2025 edition, authored by:

* [Berta Carner](https://www.linkedin.com/in/berta-carner/)
* [Anita Quevedo](https://www.linkedin.com/in/aquevedos91/)
* [Carlos Morales](https://www.linkedin.com/in/carlos-morales-galvez/)
* [Esteve Graells](https://www.linkedin.com/in/egraells/)


Advised by professor [Mariona](https://www.linkedin.com/in/marionacaros/)

## Table of Contents <a name="toc"></a>

1. [Introduction](#intro)
    1. [Motivation](#motivation)
    2. [Milestones](#milestones)
2. [The data set](#datasets)
3. [Working Environment](#working_env)
4. [General Architecture](#architecture)
    1. [Main hyperparameters](#mainhyperp)
    2. [Metrics and loss criterions](#metricsandloss)
5. [Preliminary Tests](#preliminary)
    1. [First steps](#initial)
    2. [Accessing the dataset](#datasetaccess)
    3. [Finding the right parameters](#parameters)
    4. [Does the discriminator help?](#nodiscriminator)
6. [The quest for improving the results](#improvingresults)
    1. [Increasing the pixel resolution of images](#increaseresolution)
        1. [Mid resolution](#midresolution)
        3. [High resolution](#highresolution)
    2. [Instance Normalization](#instancenorm)
    3. [Data filtering](#datafiltering)
    4. [VGG Loss](#vggloss)
    5. [Using the ReduceLROnPlateau scheduler](#plateau)
7. [Quality metrics](#qualitymetrics)
    1. [Fréchet Inception Distance](#frechet)
8. [The Google Cloud instance](#gcinstance)
9. [Conclusions and Lessons Learned](#conclusions)
10. [Next steps](#next_steps)