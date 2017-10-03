package org.jwvictor.flinktrl.operators

/**
  *     Copyright 2017 Jason Victor
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

import breeze.linalg._
import org.jwvictor.flinktrl.math.MachineLearningUtilities.SomeVector


trait FtrlStringTokenizer {
  /**
    * Split a string into a sequence of tokens (strings)
    *
    * @param input a string
    * @return tokens
    */
  def split(input: String): Seq[String]
}

case object BasicStringSplitter extends FtrlStringTokenizer {
  val PUNCTUATION = "?.,!@#$%^&*()-_=+[]{};:'\"/?><"
  val NUMBERS = "1234567890"
  val WHITESPACE = "\n\t"

  override def split(input: String): Seq[String] = {
    val toRemove = PUNCTUATION + NUMBERS + WHITESPACE
    val sanitized = input.filter(!toRemove.contains(_))
    val tokens = sanitized.split(" ")
    tokens
  }
}

object TextInputOperators {
  /**
    * Tokenizes input text and applies hashing to generate a sparse vector
    *
    * @param text         the input text
    * @param outDimension the output vector dimension
    * @param tokenizer    a tokenizer
    * @return a sparse vector of counts
    */
  def textToHashVector(text: String, outDimension: Int, tokenizer: FtrlStringTokenizer): SomeVector = {
    val hashes = tokenizer.split(text).map(x => math.abs(x.hashCode) % outDimension)
    var sparse = SparseVector.zeros[Double](outDimension)
    for (i <- hashes) {
      sparse(i) = sparse(i) + 1.0
    }
    Left(sparse)
  }
}
