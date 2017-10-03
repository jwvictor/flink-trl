package org.jwvictor.flinktrl.math

import breeze.linalg.{DenseVector, SparseVector, VectorLike}

object MachineLearningUtilities {

  type MLBasicType = Double
  type SomeVector = Either[SparseVector[MLBasicType], DenseVector[MLBasicType]]

  /**
    * A sequence of observed values
    *
    * @param values a sparse or dense vector
    */
  case class ObservedValues(values: SomeVector)

  /**
    * Learned weights
    *
    * @param values a sparse or dense vector
    */
  case class LearnedWeights(values: SomeVector)

  trait FtrlSample

  /**
    * An observation
    *
    * @param values an `ObservedValues` object
    */
  case class Observation(values: ObservedValues) extends FtrlSample

  /**
    * An observation with an outcome
    *
    * @param inputValues an `ObservedValues` object representing the `X` variables
    * @param outcome     an `ObservedValues` object representing the observed outcome
    */

  case class ObservationWithOutcome(inputValues: ObservedValues, outcome: ObservedValues) extends FtrlSample

  /**
    * An observation plus the current weights: all that is needed as input to FTRL.
    *
    * @param observation
    * @param currentWeights
    */
  case class FtrlObservation(observation: ObservationWithOutcome, currentWeights: LearnedWeights) extends FtrlSample

  type ObservationOrWeights = Either[ObservationWithOutcome, LearnedWeights]

}

/**
  * Parameters to a FTRL model
  *
  * @param alpha
  * @param beta
  * @param lambda1
  * @param lambda2
  * @param numDimensions
  */
case class FtrlParameters(alpha: Double, beta: Double, lambda1: Double, lambda2: Double, numDimensions: Int)
