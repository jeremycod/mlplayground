package ml.playground

import scala.util.Random

case class NeuronLayer(
    numNeurons: Int,
    bias: Option[Double],
    neurons: List[Neuron]
) {
  def inspect(): Unit = {
    println(s"Neurons: ${neurons.length}")
    for (n <- neurons.indices) {
      println(s" Neuron $n")
      for (w <- neurons(n).weights.indices) {
        println(s"  Weight: ${neurons(n).weights(w)}")
      }
      println(s"  Bias: ${neurons(n).bias}")
    }
  }
}

object NeuronLayer {
  def apply(numNeurons: Int, bias: Option[Double]): NeuronLayer = {
    val neurons = List.fill(numNeurons)(
      Neuron(bias.getOrElse(Random.nextDouble()), List(), List(), 0.0)
    )
    new NeuronLayer(numNeurons, bias, neurons)
  }
  def feedForward(
      neuronLayer: NeuronLayer,
      inputs: List[Double]
  ): NeuronLayer = {
    neuronLayer.copy(neurons =
      neuronLayer.neurons.map(neuron => Neuron.calculateOutput(neuron, inputs))
    )
  }
}
