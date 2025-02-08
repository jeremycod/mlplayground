package ml.playground

import scala.util.Random

case class NeuralNetwork(
    numInputs: Int,
    hiddenLayer: NeuronLayer,
    outputLayer: NeuronLayer
) {
  def inspect(point: String): Unit = {
    println(s"------${point}")
    println(s"* Inputs: $numInputs")
    println("------")
    println("Hidden Layer")
    hiddenLayer.inspect()
    println("------")
    println("* Output Layer")
    outputLayer.inspect()
    println("------")
  }
}

object NeuralNetwork {

  val LEARNING_RATE = 0.5
  def apply(
      numInputs: Int,
      numHidden: Int,
      numOutputs: Int,
      hiddenLayerWeights: Option[List[Double]] = None,
      hiddenLayerBias: Option[Double] = None,
      outputLayerWeights: Option[List[Double]] = None,
      outputLayerBias: Option[Double] = None
  ): NeuralNetwork = {
    val hiddenLayer = initWeightsFromInputToLayerNeurons(
      numInputs,
      hiddenLayerWeights,
      NeuronLayer(numHidden, hiddenLayerBias)
    )
    val outputLayer = initWeightsFromInputToLayerNeurons(
      numInputs,
      outputLayerWeights,
      NeuronLayer(numOutputs, outputLayerBias)
    )
    NeuralNetwork(numInputs, hiddenLayer, outputLayer)
  }
  private def initWeightsFromInputToLayerNeurons(
      numInputs: Int,
      layerWeights: Option[List[Double]],
      neuronLayer: NeuronLayer
  ): NeuronLayer = {
    def updateWeights(
        neurons: List[Neuron],
        weights: List[Double],
        weightNum: Int
    ): List[Neuron] = {
      neurons.zipWithIndex.map { case (neuron, index) =>
        val updatedWeights = (0 until numInputs).foldLeft(neuron.weights) {
          (acc, ind) =>
            acc :+ (if (weights.isEmpty) Random.nextDouble()
                    else weights(ind + weightNum + index * numInputs))
        }
        neuron.copy(weights = updatedWeights)
      }
    }

    val updatedNeurons =
      updateWeights(neuronLayer.neurons, layerWeights.getOrElse(List()), 0)
    neuronLayer.copy(neurons = updatedNeurons)
  }

  def train(
      network: NeuralNetwork,
      trainingInputs: List[Double],
      trainingOutputs: List[Double]
  ): NeuralNetwork = {

    // 1. FeedForward
    val hiddenLayer =
      NeuronLayer.feedForward(network.hiddenLayer, trainingInputs)
    val outputLayer = NeuronLayer.feedForward(
      network.outputLayer,
      hiddenLayer.neurons.map(_.output)
    )

    // 2. Output neuron deltas
    val pdErrorsWrtOutputNeuronTotalNetInput = {
      outputLayer.neurons.indices.map { i =>
        val neuron = outputLayer.neurons(i)
        Neuron.calculatePdErrorWrtTotalNetInput(
          neuron,
          trainingOutputs(i)
        )
      }
    }

    // 3. Hidden neuron deltas
    val pdErrorsWrtHiddenNeuronTotalNetInput = hiddenLayer.neurons.indices.map {
      i =>
        val neuron = hiddenLayer.neurons(i)
        val outputLayerWeights = outputLayer.neurons.map(_.weights(i))
        val outputLayerDeltas = outputLayer.neurons.indices.map { j =>
          pdErrorsWrtOutputNeuronTotalNetInput(j) * outputLayerWeights(j)
        }
        val pdErrorWrtTotalNetInput =
          outputLayerDeltas.sum * Neuron.calculatePdTotalNetInputWrtInput(
            neuron.output
          )
        pdErrorWrtTotalNetInput
    }

    // 4. Update output neuron weights
    val updatedOutputLayer =
      outputLayer.copy(neurons = outputLayer.neurons.zipWithIndex.map {
        case (neuron, i) =>
          val updatedWeights =
            neuron.weights.zipWithIndex.map { case (weight, j) =>
              val pdErrorWrtWeight =
                pdErrorsWrtOutputNeuronTotalNetInput(i) * neuron.inputs(j)
              weight - LEARNING_RATE * pdErrorWrtWeight
            }
          neuron.copy(weights = updatedWeights)
      })

    // 4. Update hidden neuron weights
    val updatedHiddenLayer =
      hiddenLayer.copy(neurons = hiddenLayer.neurons.zipWithIndex.map {
        case (neuron, i) =>
          val updatedWeights =
            neuron.weights.zipWithIndex.map { case (weight, j) =>
              val pdErrorWrtWeight =
                pdErrorsWrtHiddenNeuronTotalNetInput(i) * neuron.inputs(j)
              weight - pdErrorWrtWeight
            }
          neuron.copy(weights = updatedWeights)
      })
    network.copy(
      hiddenLayer = updatedHiddenLayer,
      outputLayer = updatedOutputLayer
    )
  }

  def calculateTotalError(
      network: NeuralNetwork,
      trainingSets: List[(List[Double], List[Double])]
  ): Double = {
    val totalError = trainingSets.map {
      case (trainingInputs, trainingOutputs) =>
        val hiddenLayer =
          NeuronLayer.feedForward(network.hiddenLayer, trainingInputs)
        val outputLayer = NeuronLayer.feedForward(
          network.outputLayer,
          hiddenLayer.neurons.map(_.output)
        )
        val trainingSetTotalError = outputLayer.neurons.map { neuron =>
          Neuron.calculateError(
            neuron.output,
            trainingOutputs(outputLayer.neurons.indexOf(neuron))
          )
        }.sum
        trainingSetTotalError
    }
    totalError.sum
  }

}
