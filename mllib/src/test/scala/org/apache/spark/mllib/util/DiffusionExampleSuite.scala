package org.apache.spark.mllib.util

import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.FunSuite
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._

class DiffusionExampleSuite extends FunSuite with MLlibTestSparkContext  {

  test("diffusion should work") {

    val L = 150e3
    val H = 100e3
    val nx = 101
    val ny = 51
    val nt = 500

    val kappa = 1e-6
    val dx = L / (nx - 1)
    val dy = H / (ny - 1)
    val dt = Math.pow(Math.min(dx, dy),  2) / kappa / 4.0;
    val q = 0.0
    val rho = 1e-4
    val cp = 100

    val diffTrans = new DiffusionTransformer(kappa, dt, dx, dy, q, rho, cp)

    val phaseSpace = Array(
      Array(0.0,1.0,4.0),
      Array(2.0,3.0,4.0),
      Array(4.2,4.2,1.3)
    )

    val inputVecForMiddleCell = Vectors.dense(Array(phaseSpace(1)(1), phaseSpace(1)(2), phaseSpace(1)(0), phaseSpace(2)(1), phaseSpace(0)(1)))

    val newMiddleCell = diffTrans.transform(inputVecForMiddleCell)

    assert(newMiddleCell ~== Vectors.dense(1.0) absTol 1e-3)

  }

}
