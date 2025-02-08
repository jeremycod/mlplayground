package ml.playground

object SimpleBackpropagationExercise {
  def main(args: Array[String]): Unit = {
    val nn = NeuralNetwork(
      2,
      2,
      2,
      hiddenLayerWeights = Some(List(0.15, 0.2, 0.25, 0.3)),
      hiddenLayerBias = Some(0.35),
      outputLayerWeights = Some(List(0.4, 0.45, 0.5, 0.55)),
      outputLayerBias = Some(0.6)
    )
    nn.inspect("Initial")
    val finalNN = (0 until 1000).foldLeft(nn) { (currentNN, i) =>
      val updatedNN =
        NeuralNetwork.train(currentNN, List(0.05, 0.1), List(0.01, 0.99))
      println(
        s"$i ${NeuralNetwork.calculateTotalError(updatedNN, List((List(0.05, 0.1), List(0.01, 0.99))))}"
      )
      updatedNN
    }
    finalNN.inspect("Final")

  }
}
