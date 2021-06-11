# E4C Consumptiom Forecast

## Table of Contents
* [Authors and Advisor](#authors-and-advisor)
* [Introduction](#introduction)
* [Technologies Used](#technologies-used)

## Authors and Advisor

Gabriel Pires Sobreira, Student at Ensta Paris, Institut Polytechnique de Paris

Pedro Branco, Student at Ensta Paris, Institut Polytechnique de Paris

**Adivisor**: Prof. Hossam Afifi (Telecom SudParis, Institut Polytechnique de Paris)

## Introduction

With the growth of smart grids there is a necessity to develop new techniques to improve their performance. As a result to better menage the usage of batterie and electricity it is of great interest to have a prediction of the consumption for the next hours so the Smart Grid can better otimizate it self resulting in lower energy bills. Consequently,  this project implements a RNN using LSTMs cells to do the forecast of energy demand for each hour in a window of 24 hours in the future. To do so it was used to train the model a real data set from the DrahiX building witch belongs to the École Polytechnique. This data set contains not only the consumption data, but also some weather data, such as temperature and reltive humidity, as the system also uses solar energy and this could auxiliate the RNN model to have a better accuracy.

## Technologies Used

Language used:

    Python version 3.6.9
    jupyter-notebook version 6.4.0

Libraries used:

    matplotlib version:  3.3.4
    pandas version:  1.1.5
    sklearn version:  0.24.2
    numpy version:  1.19.5
    tensorflow version:  1.15.0

Operational system used:

    Description:	Ubuntu 18.04.5 LTS
    Release:	18.04
    Codename:	bionic

It is highly recommend to use the same versions as described above to aviod any compatibility issues. Nervertheless the only library that must be the same version it is the Tensorflow, and also must be used any python 3 version.  
