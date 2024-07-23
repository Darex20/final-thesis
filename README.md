# Predicting Stock Market Movements Using Neural Networks

This repository contains the code and resources for the thesis titled "Predicting Stock Market Movements Using Neural Networks" by Dario Pavlović. The thesis explores the application of machine learning, specifically neural networks, to predict stock market prices.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

This project aims to predict stock market movements using deep learning techniques, particularly Long Short-Term Memory (LSTM) neural networks. The motivation behind this research is to explore whether stock prices follow a discernible pattern that can be captured and predicted by neural networks.

## Usage

1. **Data Preparation**: Download historical stock price data using Yahoo Finance API. Save the data in the `data/` directory.
   
2. **Run Notebooks**: Execute the Jupyter notebooks in the `notebooks/` directory to preprocess the data, train the models, and evaluate the results. Each notebook contains detailed instructions and explanations.

3. **Model Training**: Train the LSTM model using the prepared datasets. The training process involves setting various hyperparameters like `epochs`, `timestep`, and `batch_size`.

4. **Evaluation**: Evaluate the trained model using the test dataset and visualize the predictions.

## Results

The results of the thesis show that while LSTM models can capture some patterns in stock prices for short-term predictions, their overall performance is hindered by the random nature and external influences affecting stock prices. The model's performance can be visualized through the plots generated in the `results/` directory.

## Conclusion

The thesis concludes that stock prices do not follow a predictable pattern that can be reliably captured using neural networks without considering external factors. LSTM models perform better for short-term predictions but struggle with long-term accuracy due to the inherent randomness in stock price movements.

## References

1. Čupić, Marko. *Uvod u strojno učenje*. Zagreb, 2020.
2. Kingma, D. P., Ba, J. *Adam: A method for stochastic optimization*, 2014.
3. Hardt, M., Recht, B., Singer, Y. *Train faster, generalize better: Stability of stochastic gradient descent*, International Conference on Machine Learning, 2016.
4. [IBM: Recurrent Neural Networks](https://www.ibm.com/cloud/learn/recurrent-neural-networks)
5. [Andrej Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
6. [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
7. [Investopedia: Efficient Market Hypothesis](https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp)
8. [Towards Data Science: Predicting Stock Prices with a Neural Network](https://towardsdatascience.com/is-it-possible-to-predict-stock-prices-with-a-neural-network-d750af3de50b)

For a detailed explanation of the methods and results, please refer to the full thesis document.
