package org.apache.spark.mllib.feature

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.rdd.RDD

@DeveloperApi
trait MatrixTransformer {

  def transform(matrix: Matrix): Matrix
  def transform(data: RDD[Matrix]) : RDD[Matrix] = {
    data.map(x => this.transform(x))
  }
  def transform(data: JavaRDD[Matrix]): JavaRDD[Matrix] = {
    transform(data.rdd)
  }
}
