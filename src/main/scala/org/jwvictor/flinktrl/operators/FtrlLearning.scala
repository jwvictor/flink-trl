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

import breeze.linalg.{DenseVector, SparseVector}
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.streaming.api.functions.co.CoProcessFunction
import org.apache.flink.streaming.api.scala._
import org.apache.flink.streaming.api.windowing.assigners.{ProcessingTimeSessionWindows, TumblingProcessingTimeWindows}
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.windowing.triggers.CountTrigger
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.util.Collector
import org.jwvictor.flinktrl.math.FtrlParameters
import org.jwvictor.flinktrl.math.FtrlUtilities.NaiveFtrlHeuristics
import org.jwvictor.flinktrl.math.MachineLearningUtilities._

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
  implicit class FtrlLearningStream(in: DataStream[FtrlObservation])(implicit ftrlParameters: FtrlParameters) {
    /**
      * Takes a set of observed outcomes and trains a model
      *
      * @return data stream of learned weights
      */
    def withFtrlLearning: DataStream[LearnedWeights] = {

      // Move this into function scope for closure cleaner
      val dimensions = ftrlParameters.numDimensions

      // Main update stream, flat mapped over dimensions to produce an (i, obs) pair for each dimension i.
      // Executes the math.
      // Requires a stateful operation, hence the keying.
      val allUpdates = in.flatMap { updateInput =>
        0.until(dimensions).map(i => (i, updateInput))
      }.keyBy(_._1).mapWithState((tup, state: Option[Tuple2[Int, Double]]) => {
        val t_i = state.map(_._1).getOrElse(1)
        val idx = tup._1
        val observationWithOutcome = tup._2
        // Math goes here
        val newZ_i = scala.util.Random.nextDouble()
        // End math
        ((t_i, idx, newZ_i), Some((t_i + 1) % Int.MaxValue, newZ_i))
      })

      // Groups updates by their generation.
      val updatesByGeneration = allUpdates.
        keyBy(_._1).
        window(ProcessingTimeSessionWindows.withGap(Time.minutes(3))). // Reaching the gap is a fail case...
        trigger(CountTrigger.of(dimensions)). // ... the trigger should always close the window before then.
        apply[List[Tuple2[Int, Double]]]((hashCode: Int, timeWindow: TimeWindow, sq: Iterable[(Int, Int, Double)], coll: Collector[List[Tuple2[Int, Double]]]) => {
        val data = sq.map(t => (t._2, t._3)).toList
        coll.collect(data)
      })

      // The final stream of dense weight vectors
      val weightVectorStream = updatesByGeneration.map(listIdxs => {
        var vec = DenseVector.zeros[MLBasicType](dimensions)
        listIdxs.foreach(tup => vec(tup._1) = tup._2)
        LearnedWeights(Right(vec))
      })

      // `weightVectorStream` is a `DataStream[LearnedWeightings]`
      weightVectorStream
    }
  }

  /**
    * Joins data together with learned weights to continuously update the model
    *
    * @param ftrlParameters parameters to the FTRL model
    */
  class FtrlInputJoinStream(ftrlParameters: FtrlParameters)
    extends CoProcessFunction[ObservationWithOutcome, LearnedWeights, FtrlObservation]
      with NaiveFtrlHeuristics {

    private var lastsRecordedWeights: Option[LearnedWeights] = None

    /**
      * Generates initial weights
      *
      * @return dense vector
      */
    private def generateInitWeights: DenseVector[Double] = generateInitialWeights(ftrlParameters.numDimensions)

    /**
      * On any arrival of an observation, emit the observation, using generated weights if necessary.
      *
      * @param value observation
      * @param ctx   context
      * @param out   collector
      */
    override def processElement1(value: _root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.ObservationWithOutcome, ctx: _root_.org.apache.flink.streaming.api.functions.co.CoProcessFunction[_root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.ObservationWithOutcome, _root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.LearnedWeights, _root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.FtrlObservation]#Context, out: _root_.org.apache.flink.util.Collector[_root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.FtrlObservation]): Unit = {
      // Until the first set of weights arrives, we will use a vector generated using the mixed-in `FtrlHeuristics` trait.
      val weights = synchronized {
        lastsRecordedWeights match {
          case Some(w: LearnedWeights) => w
          case _ =>
            val gw = LearnedWeights(Right(generateInitialWeights(ftrlParameters.numDimensions)))
            lastsRecordedWeights = Some(gw)
            gw
        }
      }

      // Output a pair in every case
      val input = FtrlObservation(value, weights)
      out.collect(input)
    }

    /**
      * On arrival of new weights, update the state
      *
      * @param value
      * @param ctx
      * @param out
      */
    override def processElement2(value: _root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.LearnedWeights, ctx: _root_.org.apache.flink.streaming.api.functions.co.CoProcessFunction[_root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.ObservationWithOutcome, _root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.LearnedWeights, _root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.FtrlObservation]#Context, out: _root_.org.apache.flink.util.Collector[_root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.FtrlObservation]): Unit = {
      lastsRecordedWeights = Some(value)
    }

  }

  /**
    * Uses the custom operator to create a stream of `FtrlLearning` from streams of observations and weights.
    *
    * @param in
    * @param learnedWeights
    * @param rig
    */
  class FtrlFeedbackOperator(in: DataStream[ObservationWithOutcome], learnedWeights: DataStream[LearnedWeights], rig: FtrlParameters) {

    /**
      * Create feedback stream ready for ingestion by FTRL. Takes in observation and latest weight streams, and produces
      * a stream of new weights.
      *
      * Implementation recommendation: use a mechanism like Kafka to stream weights to and from and create this feedback
      * loop, as in the example program.
      *
      * @param s1
      * @param s2
      * @return joined stream
      */
    def createFeedbackLoop(s1: DataStream[ObservationWithOutcome], s2: DataStream[LearnedWeights]): DataStream[FtrlObservation] = {
      val connected: ConnectedStreams[ObservationWithOutcome, LearnedWeights] = s1.connect[LearnedWeights](s2)
      val outStream = connected.process(new FtrlInputJoinStream(null))
      outStream
    }
  }

}
