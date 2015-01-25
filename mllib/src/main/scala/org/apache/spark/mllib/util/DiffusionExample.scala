package org.apache.spark.mllib.util

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

import scala.collection.immutable


class DiffusionExample  {


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


  //seems like a better way to do this is to keep the index in the RDD, then after doing a transform:
  //  - re-aggregate the each (idx,Vector) to make Vector be the new values used for computation
  
  def runDiffusionOnceParallel(sc: SparkContext, m: IndexedSeq[IndexedSeq[Double]]): RDD[Vector] = {


    val diffTrans = new DiffusionTransformer(kappa, dt, dx, dy, q, rho, cp)

    val maxY = m.size - 1
    val maxX = m.head.size - 1

    val updatedValsRdd: immutable.IndexedSeq[immutable.IndexedSeq[Vector]] = (0 to maxY) map {
      y =>
        (0 to maxX) map {
          x =>

            if((x == 0) || (x == maxX) || (y == 0) || (y == maxY))
              Vectors.dense(m(x)(y))
            else
              Vectors.dense(Array(m(x)(y), m(x)(y+1),m(x)(y-1),m(x+1)(y),m(x-1)(y)))
        }
    }

    val flats: RDD[Vector] = sc.makeRDD(updatedValsRdd.toArray.flatMap(v => v.toArray))

    diffTrans.transform(flats)

  }
  
  def runDiffusionOnceNaive(m: IndexedSeq[IndexedSeq[Double]]): IndexedSeq[IndexedSeq[Double]] = {

    val diffTrans = new DiffusionTransformer(kappa, dt, dx, dy, q, rho, cp)

    val maxY = m.size - 1
    val maxX = m.head.size - 1

    val updatedVals: immutable.IndexedSeq[immutable.IndexedSeq[Double]] = (0 to maxY) map {
      y =>
        (0 to maxX) map {
          x =>

            if((x == 0) || (x == maxX) || (y == 0) || (y == maxY))
              m(x)(y)
            else
              //to fully take advantage of spark, this should generate RDD[Vector], which we should fold over to get a RDD[(Vector)], which we should then transform and then somehow 'unpack'
              diffTrans.transform(Vectors.dense(Array(m(x)(y), m(x)(y+1),m(x)(y-1),m(x+1)(y),m(x-1)(y))))(0)
        }

    }

    updatedVals

  }

  def runDiffusionUntilConverged(initM: IndexedSeq[IndexedSeq[Double]]): IndexedSeq[IndexedSeq[Double]] = {

    val MAX = 100

    var m1 = initM
    var m2 = runDiffusionOnceNaive(initM)
    var iters = 1
    while((! converged(m1, m2)) || (iters > MAX)) {
      m1 = m2
      m2 = runDiffusionOnceNaive(m2)
      iters = iters + 1
    }
    m2

  }

  def converged(m1: IndexedSeq[IndexedSeq[Double]], m2: IndexedSeq[IndexedSeq[Double]]): Boolean = {
    def vectorsSameWithinTolerance(v1: Vector, v2: Vector, tol: Double): Boolean = {
      if(v1.size != v2.size) {log.error(s"attempted to compare non-equal sized vectors: $v1 $v2"); return false}

      (0 to v1.size-1).foreach(i => if( math.abs(v1.apply(i) - v2.apply(i)) > tol ) return false )
      true
    }
    m1.zip(m2).map{case (v1, v2) => vectorsSameWithinTolerance(Vectors.dense(v1.toArray), Vectors.dense(v2.toArray), 1e-3)}.foldLeft(true)(_ && _)

  }

}
