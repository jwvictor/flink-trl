package org.jwvictor.flinktrl.operators

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
    val tokens = input.split(" ")
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
