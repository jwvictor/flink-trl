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

import java.nio.ByteBuffer

import breeze.linalg.{DenseVector, SparseVector, VectorLike}

import scala.util.Try

object MachineLearningUtilities {

  type MLBasicType = Double
  type SomeVector = Either[SparseVector[MLBasicType], DenseVector[MLBasicType]]
  val BYTES_IN_DOUBLE = 8

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
  case class LearnedWeights(values: SomeVector) {
    /**
      * Dimension
      *
      * @return dimension
      */
    def length: Int = {
      values match {
        case Left(sv) => sv.length
        case Right(dv) => dv.length
      }
    }

    /**
      * Convert to array
      *
      * @return array of values
      */
    def asArray: Array[Double] = {
      values match {
        case Left(sv) => sv.toArray
        case Right(dv) => dv.toArray
      }
    }

    /**
      * Serialize to a byte buffer
      *
      * @return byte buffer
      */
    def serialize: ByteBuffer = {
      val x = 1.0
      val buf = ByteBuffer.allocate(BYTES_IN_DOUBLE * length)
      val get = asArray
      for (i <- 0.until(length)) {
        buf.putDouble(get(i))
      }
      buf
    }

  }

  /**
    * Companion object for `LearnedWeights`
    */
  object LearnedWeights {
    /**
      * Deserialize a `LearnedWeights` object from raw bytes
      *
      * @param input bytes
      * @return weights
      */
    def deserialize(input: ByteBuffer): LearnedWeights = {
      var lst: List[Double] = Nil
      while (input.hasRemaining) {
        Try {
          val d = input.getDouble
          lst ::= d
        }
      }
      val data = lst.reverse.toArray
      LearnedWeights(Right(DenseVector[Double](data)))
    }
  }

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
