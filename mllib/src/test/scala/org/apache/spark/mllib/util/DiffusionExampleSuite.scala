package org.apache.spark.mllib.util

import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.FunSuite
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.mllib.rdd.RDDFunctions._
import org.slf4j.LoggerFactory

import scala.collection.immutable


class DiffusionExampleSuite extends FunSuite with MLlibTestSparkContext  {

  val log = LoggerFactory.getLogger(getClass)


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

  //phase space includes BCs, so this is once diffusion cell:
  val M1 = Array(
    Array(0.0,1.0,4.0),
    Array(2.0,3.0,4.0),
    Array(4.2,4.2,1.3)
  )

  val M2 = IndexedSeq(
    IndexedSeq(1.0,1.0,1.0,1.0),
    IndexedSeq(1.0,0.0,0.0,1.0),
    IndexedSeq(1.0,0.0,0.0,1.0),
    IndexedSeq(1.0,1.0,1.0,1.0)
  )




  test("diffusion should work for one middle cell") {


    val diffTrans = new DiffusionTransformer(kappa, dt, dx, dy, q, rho, cp)

    val inputVecForMiddleCell = Vectors.dense(Array(M1(1)(1), M1(1)(2), M1(1)(0), M1(2)(1), M1(0)(1)))

    val newMiddleCell = diffTrans.transform(inputVecForMiddleCell)

    assert(newMiddleCell ~== Vectors.dense(2.8875) absTol 1e-3)
    
  }
  
  test("diffusion for a complete phase space") {

    val diffTrans = new DiffusionTransformer(kappa, dt, dx, dy, q, rho, cp)

    val maxY = M2.size - 1
    val maxX = M2.head.size - 1

    val updatedVals: immutable.IndexedSeq[immutable.IndexedSeq[Double]] = (0 to maxY) map {
        y =>
        (0 to maxX) map {
          x =>

            if((x == 0) || (x == maxX) || (y == 0) || (y == maxY))
              M2(x)(y)
            else
              diffTrans.transform(Vectors.dense(Array(M2(x)(y), M2(x)(y+1),M2(x)(y-1),M2(x+1)(y),M2(x-1)(y))))(0)
        }

    }

//    assert(updatedVals.toArray === Array(""))
  }

  test("runs until conv") {
    val de = new DiffusionExample
    val fin = de.runDiffusionUntil(M2)
    fin.foreach(v => Vectors.dense(v.toArray) ~== Vectors.dense(1.0,1.0,1.0,1.0) absTol 1e-3)
  }

}
