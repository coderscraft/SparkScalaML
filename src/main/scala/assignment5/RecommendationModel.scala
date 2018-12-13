package assignment5
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions._

import org.apache.spark.mllib.evaluation.RankingMetrics

import scala.collection.mutable.ListBuffer


/**
  * Created by ravirane on 11/28/18.
  *
  * time spark-submit --class assignment5.RecommendationModel --master yarn-client --num-executors 5
  * --driver-memory 512m --executor-memory 512m --executor-cores 5
  * ./sparkcs657.jar /user/rrane/data/D20M/rating.csv /user/rrane/data/movies.csv /user/rrane/data/D20M/
  */
object RecommendationModel {

  // Schema for Rating file
  val customSchema = StructType(Array(
    StructField("userId", LongType, true),
    StructField("movieId", LongType, true),
    StructField("rating", FloatType, true),
    StructField("timestamp", LongType, true)))


  def main(args: Array[String]): Unit = {

    // Creating spark session
     val spark = SparkSession.builder()
      //.config("spark.master", "local[2]")
      .appName("Movies Recommendation Model")
      .getOrCreate()

    import spark.implicits._

    // Taking input for Rating/Movie and output files
    val ratingFile = args(0)
    val moviesFile = args(1)
    val outPath = args(2)

    // Reading Rating and movie files in Dataframe
    val ratingsDF = spark.read.format("com.databricks.spark.csv")
      .option("header", true)
      .schema(customSchema)
      .load(ratingFile)

    val df2 = spark.read.format("com.databricks.spark.csv").option("header", "true").load(moviesFile)

    // Building hashmap for movie files to access moview based on movie id
    val moviesRdd = df2.select(df2.col("movieId"), df2.col("title"), df2.col("genres")).rdd
    val movieMap = moviesRdd.map(m => (m(0).toString.toInt, m(1).toString)).collectAsMap()

    // Creating traning and test split of rating data [80% Training and 20% test data]
    val splits = ratingsDF.randomSplit(Array(0.8, 0.2), seed = 12345L)
    val (trainingData, testData) = (splits(0), splits(1))
    val numTraining = trainingData.count()
    val numTest = testData.count()

    // Perform 5 fold cross validation on traning data - This returns back the FoldPerformance and Hyperparameter setting back
    val foldPerformance = perform5FoldCv(trainingData, 5)
    // Select best set of Hyperparameter based on Minimum rmse value
    var minRmse = foldPerformance(0)._2
    var itr = foldPerformance(0)._3
    var regParam = foldPerformance(0)._4
    foldPerformance.foreach((fold) => {
      if(fold._2 < minRmse) {
        minRmse = fold._2
        itr = fold._3
        regParam = fold._4
      }
    })

    // Build ALS model from 80% training data using best parameter values we received from cross validation
    val als = new ALS()
      .setMaxIter(itr)
      .setRegParam(regParam)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(trainingData)
    model.setColdStartStrategy("drop")
    // Perform prediction on 20% test data
    val predictions = model.transform(testData)


   // Steps to find out MAPusing RankingMetrics

    // Get actual movies user liked that is rating > 2.5
    val perUserActual = predictions
      .where("rating > 2.5")
      .groupBy("userId")
      .agg(expr("collect_set(movieId) as movies"))


    // Get predicted movies for all user
    val perUserPredictions = predictions
      .orderBy(col("userId"), col("prediction").desc)
      .groupBy("userId")
      .agg(expr("collect_list(movieId) as movies"))

    // Join actual and predicted movies using usedId key column and generate required RDD
    val perUserActualvPred = perUserActual.join(perUserPredictions, Seq("userId"))
      .map(row => (
        row(1).asInstanceOf[Seq[Long]].toArray,
        row(2).asInstanceOf[Seq[Long]].toArray.take(15)
      ))

    // Using required RDD generate Rank or MAP value
    val ranks = new RankingMetrics(perUserActualvPred.rdd)

    print("Rank: " + ranks.meanAveragePrecision)

    // Using prediction and actual values evaluate RMSE/MSE using RegressionEvaluator
    val evaluator1 = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator1.evaluate(predictions)

    val evaluator2 = new RegressionEvaluator()
      .setMetricName("mse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val mse = evaluator2.evaluate(predictions)

    // Store all Metric values in List to persist them later
    foldPerformance.append(("Final- rmse",rmse, itr, regParam))
    foldPerformance.append(("Final- mse",mse, itr, regParam))
    foldPerformance.append(("Final- MAP",ranks.meanAveragePrecision , itr, regParam))

    // Original data was mixed with 20 cutom ratings for User Id 0. We will get all the prediction and then filter
    // the predictions for user 0
    val userRecs = model.recommendForAllUsers(5).filter($"userId" === 0)

    // Get flat list of Movies for user 0
    val finalRecom = userRecs.selectExpr("userId", "explode(recommendations)")
    // Persist movies in flat file
    val recommendations = new ListBuffer[String]()
    val movieRecomm = finalRecom.select($"col.movieId".as("movieId")).map(row => movieMap.getOrElse(row.getAs[Int](0),"Empty"))

    movieRecomm.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true")
      .save(outPath + "recommendations")

    // Persist Fold performance and final Test performance numbers in flat file
    foldPerformance.toDF().coalesce(1).write.format("com.databricks.spark.csv").option("header", "true")
      .save(outPath + "foldPerformance")

  }


  /**
    * This method perform 5 fold cross validation on Train data
    * @param data
    * @param k
    * @return
    */
  def perform5FoldCv(data: Dataset[Row], k: Int): ListBuffer[(String, Double, Int, Double)] = {
    // Create five fold
    val weights = Array(0.2, 0.2, 0.2, 0.2, 0.2)
    val splits = data.randomSplit(weights)

    // Combine 4-1 fold to generate train and test set - total 5
    val fold1Train =  Seq(splits(0), splits(1), splits(2), splits(3)).reduce(_ union _)
    val fold1Test = splits(4)
    val fold2Train =  Seq(splits(1), splits(2), splits(3), splits(4)).reduce(_ union _)
    val fold2Test = splits(0)
    val fold3Train =  Seq(splits(2), splits(3), splits(4), splits(0)).reduce(_ union _)
    val fold3Test = splits(1)
    val fold4Train =  Seq(splits(3), splits(4), splits(0), splits(1)).reduce(_ union _)
    val fold4Test = splits(2)
    val fold5Train =  Seq(splits(4), splits(0), splits(1), splits(2)).reduce(_ union _)
    val fold5Test = splits(3)

    // Perform ALS and generate fold performance number using RegressionEvaluator - rmse
    // Used different set of hyperparameters for Number of iterations and regularization
    val foldPerformance = new ListBuffer[(String, Double, Int, Double)]()
    val rmse1 = performAls(fold1Train, fold1Test, 5, 0.01)

    foldPerformance.append(("Fold-1",rmse1, 5, 0.01))
    val rmse2 = performAls(fold1Train, fold1Test, 8, 0.1)
    foldPerformance.append(("Fold-2",rmse2, 8, 0.1))

    val rmse3 = performAls(fold1Train, fold1Test, 10, 0.01)
    foldPerformance.append(("Fold-3",rmse3, 10, 0.01))

    val rmse4 = performAls(fold1Train, fold1Test, 12, 0.1)
    foldPerformance.append(("Fold-4",rmse4, 12, 0.1))

    val rmse5 = performAls(fold1Train, fold1Test, 15, 0.01)
    foldPerformance.append(("Fold-5",rmse5, 15, 0.01))

    foldPerformance

  }

  /**
    *  Perform ALS on given data and generate rmse value and return it
    * @param train
    * @param test
    * @param maxItr
    * @param regParam
    * @return
    */
  def performAls(train: Dataset[Row], test: Dataset[Row], maxItr: Int, regParam: Double): Double = {
    val als = new ALS()
      .setMaxIter(maxItr)
      .setRegParam(regParam)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(train)

    model.setColdStartStrategy("drop")
    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)

    rmse
  }

}
