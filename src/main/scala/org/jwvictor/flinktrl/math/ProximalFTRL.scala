package org.jwvictor.flinktrl.math

/**
  * Copyright 2017 Jason Victor
  *
  * Licensed under the Apache License, Version 2.0 (the "License");
  * you may not use this file except in compliance with the License.
  * You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */

import breeze.linalg.DenseVector
import org.jwvictor.flinktrl.math.MachineLearningUtilities.{LearnedWeights, MLBasicType, SomeVector}

/**
  * Implementation of proximal Follow the Regularized Leader algorithm from Google (c.f. README).
  */
object ProximalFTRL {

  /**
    * Step output
    *
    * @param index vector index
    * @param newZ
    * @param newN
    * @param newW
    */
  case class FtrlGenerationStepOutput(index: Double, newZ: MLBasicType, newN: MLBasicType, newW: MLBasicType)


  /**
    * Sigmoid function
    *
    * @param x
    * @return
    */
  private def logisticSigmoid(x: Double): Double = {
    1.0 / (1.0 + scala.math.exp(-1.0 * x))
  }

  /**
    * Sign function
    *
    * @param x             value
    * @param zeroPrejudice treat zero as negative
    * @return
    */
  private def sgn(x: Double, zeroPrejudice: Boolean = true): Double = {
    if (x > 0.0 || (x == 0.0 && !zeroPrejudice)) 1.0 else -1.0
  }

  /**
    * Parameter update function: this is the core mathematical function of the model
    *
    * @param index          index to update
    * @param ftrlParameters parameters to model
    * @param inputVector    observed vector
    * @param label          observed outcome
    * @param learnedWeights current weights
    * @param precomputedPt  pre-computed `p_t` (for future optimizations)
    * @return
    */
  def computeNextGeneration(index: Int,
                            ftrlParameters: FtrlParameters,
                            inputVector: SomeVector,
                            label: Double,
                            learnedWeights: LearnedWeights,
                            precomputedPt: Option[Double] = None): FtrlGenerationStepOutput = {

    // Current values
    val wi = learnedWeights.values(index)
    val ni = learnedWeights.nVector(index)
    val zi = learnedWeights.zVector(index)

    // Get updated weights based on new `z_i` and `n_i` values
    val newWeights = 0.until(learnedWeights.length).map { i =>
      if (math.abs(learnedWeights.values(i)) <= ftrlParameters.lambda1) 0.0 else {
        -1.0 * math.sqrt(((ftrlParameters.beta + math.sqrt(ni)) / ftrlParameters.alpha) + ftrlParameters.lambda2) * (zi - (sgn(zi) * ftrlParameters.lambda1))
      }
    }.toArray
    val newWeightVector = DenseVector(newWeights)

    // Compute `p_t` as in the paper
    val pt = precomputedPt match {
      case Some(x) => x
      case None =>
        val input = inputVector.asSparse
        val dProd: Double = input.dot(newWeightVector)
        logisticSigmoid(dProd)
    }

    // Compute the gradient and adjust for regularization
    val gi = (pt - label) * inputVector(index)
    val sigmai = (1.0 / ftrlParameters.alpha) * (math.sqrt(ni + (gi * gi)) - math.sqrt(ni))

    // Obtain new parameters
    val newZi = zi + gi - (sigmai * newWeightVector(index))
    val newNi = ni + (gi * gi)

    // Return updated parameters
    FtrlGenerationStepOutput(index, newZi, newNi, newWeightVector(index))
  }
}
