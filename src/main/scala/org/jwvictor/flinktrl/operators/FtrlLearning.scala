package org.jwvictor.flinktrl.operators

/**
  * Licensed to the Apache Software Foundation (ASF) under one
  * or more contributor license agreements.  See the NOTICE file
  * distributed with this work for additional information
  * regarding copyright ownership.  The ASF licenses this file
  * to you under the Apache License, Version 2.0 (the
  * "License"); you may not use this file except in compliance
  * with the License.  You may obtain a copy of the License at
  *
  * http://www.apache.org/licenses/LICENSE-2.0
  *
  * Unless required by applicable law or agreed to in writing, software
  * distributed under the License is distributed on an "AS IS" BASIS,
  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  * See the License for the specific language governing permissions and
  * limitations under the License.
  */

import java.util
import java.util.Collections

import breeze.linalg.{DenseVector, SparseVector}
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.streaming.api.scala._
import org.jwvictor.flinktrl.math.FtrlParameters
import org.jwvictor.flinktrl.math.MachineLearningUtilities.{LearnedWeights, MLBasicType, ObservationWithOutcome}

/**
  * Helper types to make `Long` values implement `java.io.Serializable`
  */
object FtrlLearningTypeHelpers {

  /**
    * Wrapper type
    *
    * @param x value
    */
  case class SerializableLong(x: Long) extends Serializable

  /**
    * Implicit conversion to
    *
    * @param x
    * @return
    */
  implicit def longToSerializable(x: Long): SerializableLong = SerializableLong(x)

  /**
    * Implicit conversion from
    *
    * @param s
    * @return
    */
  implicit def serializableToLong(s: SerializableLong): Long = s.x
}

import FtrlLearningTypeHelpers._

/**
  * `FtrlLearningStream` implicit class container
  */
object FtrlLearning {

  // Declare type information to Flink type system
  implicit val typeInfo = TypeInformation.of(classOf[LearnedWeights])

  /**
    * Implicit class that adds a `withFtrlLearning` method to a `DataStream`
    *
    * @param in
    * @param ftrlParameters
    */
  implicit class FtrlLearningStream(in: DataStream[ObservationWithOutcome])(implicit ftrlParameters: FtrlParameters) {
    /**
      * Takes a set of observed outcomes and trains a model
      *
      * @return data stream of learned weights
      */
    def withFtrlLearning: DataStream[LearnedWeights] = {

      val dimensions = ftrlParameters.numDimensions
      val allUpdates = in.flatMap { updateInput =>
        0.until(dimensions).map(i => (i, updateInput))
      }.keyBy(_._1).mapWithState((tup, state: Option[Double]) => {
        val idx = tup._1
        val observationWithOutcome = tup._2
        // Math goes here
        val newZ_i = scala.util.Random.nextDouble()
        // End math
        val hashCode = observationWithOutcome.hashCode
        ((hashCode, idx, newZ_i), Some(newZ_i))
      })
      /*val updatesByHashCode = allUpdates.keyBy(_._1).
        window(ProcessingTimeSessionWindows.withGap(Time.seconds(5))).apply[List[Tuple2[Int,Double]]]((i:Int,tw:TimeWindow,sq:Iterable[(Int,Int,Double)],coll:Collector[List[Tuple2[Int,Double]]]) => {
        //window(EventTimeSessionWindows.withGap(Time.seconds(5))).apply[List[Tuple2[Int,Double]]]((i:Int,tw:TimeWindow,sq:Iterable[(Int,Int,Double)],coll:Collector[List[Tuple2[Int,Double]]]) => {
        val data = sq.map(t => (t._2, t._3)).toList
        coll.collect(data)
      })

            val weightVectorStream = updatesByHashCode.map(listIdxs => {
              var vec = DenseVector.zeros[MLBasicType](ftrlParameters.numDimensions)
              listIdxs.foreach(tup => vec(tup._1) = tup._2)
              LearnedWeights(Right(vec))
            })

            weightVectorStream
            */
      allUpdates.map(lst => LearnedWeights(Left(SparseVector.zeros[Double](lst._2))))
    }
  }

}
