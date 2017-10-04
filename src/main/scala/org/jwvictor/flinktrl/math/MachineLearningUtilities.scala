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

import java.nio.ByteBuffer

import breeze.linalg.{DenseVector, SparseVector, VectorLike}

import scala.util.Try

object MachineLearningUtilities {

  type MLBasicType = Double

  /**
    * The `SomeVector` type allows dense, sparse, and initialization vectors to be used interchangeably.
    */
  trait SomeVector {
    /**
      * Cast to simple array
      *
      * @return
      */
    def asArray: Array[MLBasicType]

    /**
      * Coerce to sparse vector
      *
      * @return
      */
    def asSparse: SparseVector[MLBasicType]

    /**
      * Coerce to dense vector
      *
      * @return
      */
    def asDense: DenseVector[MLBasicType]

    /**
      * Indexing function
      *
      * @param index
      * @return
      */
    def apply(index: Int): MLBasicType
  }

  /**
    * Implementation for a sparse vector
    *
    * @param sparseVector
    */
  case class Sparse(sparseVector: SparseVector[Double]) extends SomeVector {
    override def asArray: Array[MLBasicType] = sparseVector.toArray

    override def asSparse: SparseVector[MLBasicType] = sparseVector

    override def asDense: DenseVector[MLBasicType] = DenseVector(sparseVector.toArray)

    override def apply(index: Int): MLBasicType = sparseVector(index)

  }

  /**
    * Implementation for a dense vector
    *
    * @param denseVector
    */
  case class Dense(denseVector: DenseVector[Double]) extends SomeVector {
    override def asArray: Array[MLBasicType] = denseVector.toArray

    override def asSparse: SparseVector[MLBasicType] = SparseVector(denseVector.toArray)

    override def asDense: DenseVector[MLBasicType] = denseVector

    override def apply(index: Int): MLBasicType = denseVector(index)
  }

  /**
    * Implementation for an initialization (zero) vector
    */
  case object Initialization extends SomeVector {
    override def asArray: Array[MLBasicType] = Array()

    override def asSparse: SparseVector[MLBasicType] = SparseVector.zeros[Double](0)

    override def asDense: DenseVector[MLBasicType] = DenseVector.zeros[Double](0)

    override def apply(index: Int): MLBasicType = 0
  }

  val BYTES_IN_DOUBLE = 8

  /**
    * A sequence of observed values
    *
    * @param values a sparse or dense vector
    */
  case class ObservedValues(values: SomeVector)

  implicit def observedValuesToVector(o: ObservedValues): SomeVector = o.values

  /**
    * Learned weights
    *
    * @param values a sparse or dense vector
    */
  case class LearnedWeights(values: SomeVector, zVector: SomeVector, nVector: SomeVector) {
    /**
      * Dimension
      *
      * @return dimension
      */
    def length: Int = {
      values match {
        case Sparse(sv) => sv.length
        case Dense(dv) => dv.length
        case _ => 0
      }
    }

    /**
      * Convert to array
      *
      * @return array of values
      */
    def asArray: Array[Double] = {
      values.asArray
    }

    /**
      * Serialize to a byte buffer
      *
      * @return byte buffer
      */
    def serialize: ByteBuffer = {
      val x = 1.0
      val buf = ByteBuffer.allocate(BYTES_IN_DOUBLE * length * 3)
      val data = asArray
      for (i <- 0.until(length)) {
        buf.putDouble(data(i))
      }
      val zv = zVector.asArray
      for (i <- 0.until(length)) {
        buf.putDouble(zv(i))
      }
      val nv = nVector.asArray
      for (i <- 0.until(length)) {
        buf.putDouble(nv(i))
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
      val len = data.length / 3
      assert((len % 3) == 0)

      val ws = data.slice(0, len)
      val zs = data.slice(len, 2 * len)
      val ns = data.slice(2 * len, 3 * len)
      LearnedWeights(Dense(DenseVector[Double](ws)), Dense(DenseVector[Double](zs)), Dense(DenseVector[Double](ns)))
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
