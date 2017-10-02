package org.jwvictor.flinktrl

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

import org.apache.flink.streaming.api.scala._
import org.jwvictor.flinktrl.operators.{BasicStringSplitter, TextInputOperators}
import breeze.linalg._
import org.jwvictor.flinktrl.math.FtrlParameters
import org.jwvictor.flinktrl.math.MachineLearningUtilities.{ObservationWithOutcome, ObservedValues}


/**
  * Entry point for test driver.
  */
object Job {
  def main(args: Array[String]) {
    // Set up the execution environment
    val env = StreamExecutionEnvironment.getExecutionEnvironment
    env.setMaxParallelism(1)

    import org.apache.flink.streaming.api.scala._


    // Basic model parameters
    val nDimensions = 100
    implicit val ftrlParameters = FtrlParameters(1, 1, 1, 1, nDimensions) // implicit input to `withFtrlLearning
    val res = SparseVector.zeros[Double](1)
    // Input stream
    val txtStream = env.readTextFile("testdata.dat")

    import org.jwvictor.flinktrl.operators.FtrlLearning._

    // Test `withFtrlLearning` operator
    val outStream = txtStream.
      map(TextInputOperators.textToHashVector(_, nDimensions, BasicStringSplitter)).
      map(x => ObservationWithOutcome(ObservedValues(x), ObservedValues(Left(res)))).
      withFtrlLearning.
      map(_.toString)
    outStream.writeAsText("./out-ftrl-test.dat")

    env.execute("FlinkTRL test driver")
  }
}
