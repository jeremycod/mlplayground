# MLPlayground

## Overview

MLPlayground is a Scala-based project designed to help you learn and understand the backpropagation algorithm used in training neural networks. This project provides a simple implementation of a neural network and demonstrates how backpropagation can be used to train the network.

## Features

- Implementation of a simple neural network with one hidden layer
- Initialization of weights and biases
- Forward propagation to calculate outputs
- Backpropagation to update weights and biases based on the error
- Example usage to train the network on a simple dataset

## Prerequisites

- Scala 2.13.16
- sbt 1.5.5 or later

## Getting Started

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/mlplayground.git
    cd mlplayground
    ```

2. **Build the project:**

    ```sh
    sbt compile
    ```

3. **Run the example:**

    ```sh
    sbt run
    ```

## Project Structure

- `build.sbt`: SBT build configuration file
- `project/plugins.sbt`: SBT plugins configuration file
- `src/main/scala/ml/playground/SimpleBackpropagationExercise.scala`: Main entry point for the example
- `src/main/scala/ml/playground/NeuralNetwork.scala`: Implementation of the neural network and backpropagation algorithm

## Usage

The main entry point for the project is the `SimpleBackpropagationExercise` object. This object initializes a neural network, trains it using backpropagation, and prints the error at each iteration.

