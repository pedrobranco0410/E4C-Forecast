# E4C Consumptiom Forecast

## Table of Contents
* [Introduction](#introduction)
* [Project Details](#project-details)
* [Technologies Used](#technologies-used)
* [Code](#code)
    * [Notebooks](#notebooks)
    * [Codes](#codes)
    * [Server_Codes](#server_codes)
* [Authors and Advisor](#authors-and-advisor)

## Introduction

With the growth of smart grids there is a necessity to develop new techniques to improve their performance. In order to better menage the usage of energy from all the sources available it is of great interest to have a prediction of the consumption for the next hours so the Smart Grid can otimizate it self resulting in lower energy bills.

## Project Details

This project implements a model of RNN using LSTMs cells to do the forecast of energy demand for each hour in a window of 24 hours in the future. To do so it was used to train the model a real data set from the DrahiX building witch belongs to the École Polytechnique. This data set contains not only the consumption data, but also some weather data, such as temperature and reltive humidity, as the system in consideration also uses solar energy and this variables are directly linked and could auxiliate the RNN model to have a better accuracy in the forecast.

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

## Code

There are three folder with codes to run (Codes, Notebooks and Server_Codes).

### Notebooks

The folder 'Notebooks' contains the notebooks to train and test the model and executing in the jupyter notebook

### Codes

The folder 'Codes' contains the .py code to train and test model executing in the terminal.

### Server_Codes

The folder 'Server_Codes' contains the code to execute in the server to get data and make forecast in realtime using the data that is available and updated in a google drive.

## Authors and Advisor

Gabriel Pires Sobreira, Student at Ensta Paris, Institut Polytechnique de Paris

Pedro Branco, Student at Ensta Paris, Institut Polytechnique de Paris

**Adivisor**: Prof. Hossam Afifi (Telecom SudParis, Institut Polytechnique de Paris)
