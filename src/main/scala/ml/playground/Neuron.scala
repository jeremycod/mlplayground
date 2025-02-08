package ml.playground

import scala.math.{exp, pow}

case class Neuron(
    bias: Double,
    weights: List[Double],
    inputs: List[Double],
    output: Double
)

object Neuron {
  private def squash(totalNetInput: Double): Double = {
    1.0 / (1.0 + exp(-totalNetInput))
  }
  private def calculateTotalNetInput(
      inputs: List[Double],
      weights: List[Double],
      bias: Double
  ): Double = {
    inputs.zip(weights).map { case (i, w) => i * w }.sum + bias
  }
  def calculateOutput(neuron: Neuron, inputs: List[Double]): Neuron = {
    val output = squash(
      calculateTotalNetInput(inputs, neuron.weights, neuron.bias)
    )
    neuron.copy(inputs = inputs, output = output)
  }

  /** The partial derivate of the error with respect to actual output then is calculated by:
    * = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    * = -(target output - actual output)
    *
    * The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
    * = actual output - target output
    *
    * Alternative, you can use (target - output), but then need to add it during backpropagation [3]
    *
    * Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
    * = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    * @param targetOutput
    * @param output
    * @return
    */
  private def calculatePdErrorWrtOutput(
      targetOutput: Double,
      output: Double
  ): Double =
    -(targetOutput - output)

  /** The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
    * yⱼ = φ = 1 / (1 + e^(-zⱼ))
    * Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
    *
    * The derivative (not partial derivative since there is only one variable) of the output then is:
    * dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    * @param output
    * @return
    */
  def calculatePdTotalNetInputWrtInput(output: Double): Double =
    output * (1 - output)

  /** Determine how much the neuron's total input has to change to move closer to the expected output
    *
    * Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    * the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    * the partial derivative of the error with respect to the total net input.
    * This value is also known as the delta (δ) [1]
    * δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    *
    * @param neuron
    * @param targetOutput
    * @return
    */
  def calculatePdErrorWrtTotalNetInput(
      neuron: Neuron,
      targetOutput: Double
  ): Double = {
    calculatePdErrorWrtOutput(
      targetOutput,
      neuron.output
    ) * calculatePdTotalNetInputWrtInput(neuron.output)
  }

  def calculateError(targetOutput: Double, output: Double): Double = {
    0.5 * pow(targetOutput - output, 2)
  }
}
