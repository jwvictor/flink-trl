package org.jwvictor.flinktrl.math

/**
  *      Copyright 2017 Jason Victor
  *
  *  Licensed under the Apache License, Version 2.0 (the "License");
  *  you may not use this file except in compliance with the License.
  *  You may obtain a copy of the License at
  *
  *      http://www.apache.org/licenses/LICENSE-2.0
  *
  *  Unless required by applicable law or agreed to in writing, software
  *  distributed under the License is distributed on an "AS IS" BASIS,
  *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  *  See the License for the specific language governing permissions and
  *  limitations under the License.
  */

import breeze.linalg.DenseVector

/**
  * Types tangentially but not directly related to FTRL algorithm.
  */
object FtrlUtilities {

  /**
    * Defines key heuristics used in the implementation.
    */
  trait FtrlHeuristics {
    /**
      * Generates a set of initial weights based on dimensionality of the FTRL problem.
      *
      * @param nDimensions
      * @return
      */
    def generateInitialWeights(nDimensions:Int):DenseVector[Double]
  }

  /**
    * Naive heuristics:
    *  - Standard Gaussian-distributed initialization vector
    */
  trait NaiveFtrlHeuristics extends FtrlHeuristics {
    def generateInitialWeights(nDimensions:Int):DenseVector[Double] = {
      DenseVector[Double](0.until(nDimensions).map(_ => scala.util.Random.nextGaussian()).toArray)
    }
  }

}
