package org.jwvictor.flinktrl

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

import org.apache.flink.streaming.api.scala._
import org.jwvictor.flinktrl.operators.{BasicStringSplitter, TextInputOperators}
import breeze.linalg._
import breeze.numerics._
import breeze.util._
import org.apache.flink.streaming.api.functions.source.SourceFunction
import org.jwvictor.flinktrl.math.FtrlParameters
import org.jwvictor.flinktrl.math.MachineLearningUtilities.{FtrlObservation, LearnedWeights, ObservationWithOutcome, ObservedValues}


/**
  * Entry point for example driver.
  */
object Job {
  def main(args: Array[String]) {
    // Set up the execution environment
    val env = StreamExecutionEnvironment.getExecutionEnvironment
    env.setMaxParallelism(1) // TODO: get more parallelism

    import org.apache.flink.streaming.api.scala._


    // Basic model parameters
    val nDimensions = 100
    implicit val ftrlParameters = FtrlParameters(1, 1, 1, 1, nDimensions) // implicit input to `withFtrlLearning`
    val res = SparseVector.zeros[Double](1)
    // Example serialization to file
    breeze.linalg.mmwrite(new java.io.File("resoutfile.dat"), new DenseMatrix[Double](res.length, 1, res.toArray))
    // Input stream
    val fileStream = env.readTextFile("testdata.dat")
    val txtStream = env.addSource[String](new SourceFunction[String] {

      @volatile
      private var isRunning = true

      override def cancel(): Unit = {
        isRunning = false
      }

      override def run(ctx: SourceFunction.SourceContext[String]): Unit = {
        var ctr:Int = 0
        while(isRunning && ctr < 1000){
          ctx.collect(scala.util.Random.nextString(300))
          Thread.sleep(300)
          ctr += 1
        }
      }
    })

    import org.jwvictor.flinktrl.operators.FtrlLearning._

    // Test `withFtrlLearning` operator
    val outStream = txtStream.
      map(TextInputOperators.textToHashVector(_, nDimensions, BasicStringSplitter)).
      map(x => ObservationWithOutcome(ObservedValues(x), ObservedValues(Left(res)))).
      map(x => FtrlObservation(x, LearnedWeights(Right(DenseVector.zeros[Double](nDimensions))))).
      withFtrlLearning.
      map(_.toString)
    outStream.writeAsText("./out-ftrl-test.dat")

    env.execute("FlinkTRL test driver")
  }
}
