package org.apache.spark.mllib.util

import org.apache.spark.mllib.linalg.Vectors

class DiffusionExample {

  def runDiffusion(): Unit = {

    val diffTrans = new DiffusionTransformer(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    val phaseSpace = Array(
      Array(0.0,1.0,4.0),
      Array(2.0,3.0,4.0),
      Array(4.2,4.2,1.3)
    )

    val inputVecForMiddleCell = Vectors.dense(Array(phaseSpace(1)(1), phaseSpace(1)(2), phaseSpace(1)(0), phaseSpace(2)(1), phaseSpace(0)(1)))

    val newMiddleCell = diffTrans.transform(inputVecForMiddleCell)
  }

}
