package org.apache.spark.mllib.util

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.feature.VectorTransformer
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors

@Experimental
class DiffusionTransformer (kappa: Double, dt: Double, dx: Double, dy: Double, q: Double, rho: Double, cp: Double ) extends VectorTransformer{
  /**
   * Applies transformation on a vector.
   *
   * @param vector vector to be transformed.
   * @return transformed vector.
   */
  override def transform(vector: linalg.Vector): linalg.Vector = {


    require(vector.size == 5)

    val tij = vector(0)
    val tijp = vector(1)
    val tijm = vector(2)
    val tipj = vector(3)
    val timj = vector(4)

    val t_nplusone = tij + ((kappa * dt) / (dx * dx)) * (tijp - 2.0 * tij + tijm) + ((kappa * dt) / (dy * dy)) * (tipj - 2.0 * tij + timj) + (q * dt) / (rho * cp)

    Vectors.dense(t_nplusone)
  }
}
