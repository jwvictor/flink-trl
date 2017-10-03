package org.jwvictor.flinktrl.math

import breeze.linalg.DenseVector

object FtrlUtilities {

  trait FtrlHeuristics {
    def generateWeights(nDimensions:Int):DenseVector[Double]
  }
  trait NaiveFtrlHeuristics extends FtrlHeuristics {
    def generateWeights(nDimensions:Int):DenseVector[Double] = {
      DenseVector[Double](0.until(nDimensions).map(_ => scala.util.Random.nextGaussian()).toArray)
    }
  }

}
