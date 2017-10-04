package org.jwvictor.flinktrl.operators

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

import breeze.linalg.{DenseVector, SparseVector}
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.streaming.api.functions.co.CoProcessFunction
import org.apache.flink.streaming.api.scala._
import org.apache.flink.streaming.api.windowing.assigners.{ProcessingTimeSessionWindows, TumblingProcessingTimeWindows}
import org.apache.flink.streaming.api.windowing.time.Time
import org.apache.flink.streaming.api.windowing.triggers.CountTrigger
import org.apache.flink.streaming.api.windowing.windows.TimeWindow
import org.apache.flink.util.Collector
import org.jwvictor.flinktrl.math.{FtrlParameters, ProximalFTRL}
import org.jwvictor.flinktrl.math.FtrlUtilities.NaiveFtrlHeuristics
import org.jwvictor.flinktrl.math.MachineLearningUtilities._
import org.jwvictor.flinktrl.math.ProximalFTRL.FtrlGenerationStepOutput

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
      val params = ftrlParameters
      val dimensions = params.numDimensions
      import ProximalFTRL._

      // Main update stream, flat mapped over dimensions to produce an (i, obs) pair for each dimension i.
      // Executes the math.
      // Requires a stateful operation, hence the keying.
      val allUpdates = in.flatMap { updateInput =>
        0.until(dimensions).map(i => (i, updateInput))
      }.keyBy(_._1).
        mapWithState((tup, state: Option[Tuple3[Int, Double, Double]]) => {
          // The state tuple is the generation t_i, and the state variables z_i and n_i
          val t_i = state.map(_._1).getOrElse(1)
          val idx = tup._1
          val observationWithOutcome = tup._2

          // Run FTRL update computation
          val pFtrlOutput = computeNextGeneration(
            tup._1,
            params,
            observationWithOutcome.observation.inputValues,
            observationWithOutcome.observation.outcome.asArray(0),
            observationWithOutcome.currentWeights)

          // Extract relevant values
          val newZ_i = pFtrlOutput.newZ
          val newN_i = pFtrlOutput.newN
          val newW_i = pFtrlOutput.newW

          // Forward new value along and update state
          ((t_i, idx, newZ_i, newN_i, newW_i), Some((t_i + 1) % Int.MaxValue, newZ_i, newN_i))
        })

      // Groups updates by their generation.
      val updatesByGeneration = allUpdates.
        keyBy(_._1).
        window(ProcessingTimeSessionWindows.withGap(Time.minutes(3))). // Reaching the gap is a fail case...
        trigger(CountTrigger.of(dimensions)). // ... the trigger should always close the window before then.
        apply((_: Int, _: TimeWindow, sq: Iterable[(Int, Int, Double, Double, Double)], coll: Collector[(List[(Int, Double)], List[(Int, Double)], List[(Int, Double)])]) => {
        val zData = sq.map(t => (t._2, t._3)).toList
        val nData = sq.map(t => (t._2, t._4)).toList
        val wData = sq.map(t => (t._2, t._5)).toList
        coll.collect((zData, nData, wData))
      })

      // The final stream of dense weight vectors
      val weightVectorStream = updatesByGeneration.map(listIdxsTup => {
        val listIdxs = listIdxsTup._3
        var wVec = DenseVector.zeros[MLBasicType](dimensions)
        listIdxs.foreach(tup => wVec(tup._1) = tup._2)
        var zVec = DenseVector.zeros[MLBasicType](dimensions)
        listIdxsTup._1.foreach(tup => zVec(tup._1) = tup._2)
        var nVec = DenseVector.zeros[MLBasicType](dimensions)
        listIdxsTup._2.foreach(tup => nVec(tup._1) = tup._2)
        val weights = LearnedWeights(Dense(wVec), Dense(zVec), Dense(nVec))
        weights
      })

      // Return `weightVectorStream` - a `DataStream[LearnedWeights]` with updated parameters
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
      * On any arrival of an observation, emit the observation, using generated weights if necessary.
      *
      * @param value observation
      * @param ctx   context
      * @param out   collector
      */
    override def processElement1(value: _root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.ObservationWithOutcome, ctx: _root_.org.apache.flink.streaming.api.functions.co.CoProcessFunction[_root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.ObservationWithOutcome, _root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.LearnedWeights, _root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.FtrlObservation]#Context, out: _root_.org.apache.flink.util.Collector[_root_.org.jwvictor.flinktrl.math.MachineLearningUtilities.FtrlObservation]): Unit = {
      // Until the first set of weights arrives, we used to use a vector generated using the mixed-in `FtrlHeuristics` trait.
      // Now we built that logic into the math receiving the vector.
      val weights = synchronized { // Concurrent accesses aren't a problem now, but being defensive for future-proofing.
        lastsRecordedWeights match {
          case Some(w: LearnedWeights) => w
          case _ =>
            // Pass the initialization vector if none exists yet
            val gw = LearnedWeights(Initialization, Initialization, Initialization)
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
    * @param ftrlParams
    */
  implicit class FtrlFeedbackOperator(s1: DataStream[ObservationWithOutcome])(implicit ftrlParams: FtrlParameters) {

    /**
      * Create feedback stream ready for ingestion by FTRL. Takes in observation and latest weight streams, and produces
      * a stream of new weights.
      *
      * Implementation recommendation: use a mechanism like Kafka to stream weights to and from and create this feedback
      * loop, as in the example program.
      *
      * @param inputWeightStream
      * @return joined stream
      */
    def createFeedbackLoop(inputWeightStream: DataStream[LearnedWeights]): DataStream[FtrlObservation] = {
      val connected: ConnectedStreams[ObservationWithOutcome, LearnedWeights] = s1.connect[LearnedWeights](inputWeightStream)
      val outStream = connected.process(new FtrlInputJoinStream(null))
      outStream
    }
  }

}
