package assignment3

import assignment3.DecisionTree.saveToFile
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.mllib.evaluation.RegressionMetrics

/**
  * Created by ravirane on 10/28/18.
  *
  * Linear regression file will work on each leaf data of decision tree. It will ingnore the previous prediction column from
  * decision tree and generate new prediction. Here class lable is continuous trip Duration column
  */
object LRModel {

  def main(args: Array[String]): Unit = {

    /*
       Structure to hold input data files
     */
    val customSchema = StructType(Array(
      StructField("passenger_count", IntegerType, true),
      StructField("day_of_week", IntegerType, true),
      StructField("timeOfDay", IntegerType, true),
      StructField("dist_in_km", IntegerType, true),
      StructField("rain_stat", IntegerType, true),
      StructField("snow_stat", IntegerType, true),
      StructField("trip_duration", IntegerType, true)))
      //StructField("duration_bin", IntegerType, true)))
      //StructField("prediction", DoubleType, true)))

    /*
        Generate spark context object for future use
     */
    val spark = SparkSession.builder()
      //.config("spark.master", "local[2]")
      .appName("NY Taxi Regression")
      .getOrCreate()

    // Reading input file and output directory
    val trainFile = args(0)
    var outputDir = args(1)

    /*
      Sets correct name for output/result files
     */
    val splis = trainFile.split("/")
    val outFile = "regres-" + splis(splis.size -1)

    /*
       Read data file and create dataframe
     */
    val train_df = spark.read.format("csv").option("header", "true")
      .schema(customSchema)
      .load(trainFile)

    /*
       Create 80-20 split for Train and Test data
     */
    val Array(training, test) = train_df.randomSplit(Array(0.8, 0.2), seed = 12345)

    //train_df.show()

    /*
       Create array of feature columns for regression model
     */
    val featureCols = Array("passenger_count", "day_of_week", "timeOfDay", "dist_in_km", "rain_stat", "snow_stat")
    /*
       Create assembler from feature column list
     */
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    /*
       Create linear regression model with feature and class label column
     */
    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("trip_duration")
    //  .setRegParam(0.1)
    //  .setElasticNetParam(1.0)
    //  .setMaxIter(10)

    /*
       Create pipeline from Assembler and Linear Regression
     */
    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr))

    /*
         Create parameter list for cross validation
         - Regulrization parameter - 1000.0 and 0.001
         - Iterations - 1, 100
         - Tried L1,L2 using Elastic New Param but it produces bad outcome
     */
    val lrParamMaps = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(1000.0, 0.001))
      .addGrid(lr.maxIter, Array(1, 10))
      //.addGrid(lr.elasticNetParam, Array(0.05, 1.0))
      .build()

    /*
       Created Regression evaluator for class label and outcome column
     */
    var eval = new RegressionEvaluator()
      .setLabelCol("trip_duration")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
      //.setMetricName("accuracy")

    /*
       Build cross validation object with Linear Regression setup
     */
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(lrParamMaps)
      .setEvaluator(eval)
      .setNumFolds(10)

    /*
      Build linear regression model for Training data after cross validation using best outcome
     */
    val cvModel = cv.fit(training)
    /*
       Verify model on test data and generate prediction dataframe
     */
    val fullPredictions = cvModel.transform(test)
    //fullPredictions.show()
    /*
        Get prediction outcome and save it to file
     */
    val result = fullPredictions.select("passenger_count","day_of_week","timeOfDay","dist_in_km","rain_stat","snow_stat","trip_duration","prediction")
    saveToFile(result, outFile, outputDir)
    /*
       Generate performance measure for prediction
         - Root mean squared error
         - Mean absolute error
         Save it to file
     */
    //val predictions = fullPredictions.select("prediction").rdd.map(_.getDouble(0))
    //val labels = fullPredictions.select("trip_duration").rdd.map(_.getInt(0).toDouble)

    val RMSE = eval.evaluate(result)
    println(RMSE)

    eval = new RegressionEvaluator()
      .setLabelCol("trip_duration")
      .setPredictionCol("prediction")
      .setMetricName("mae")

    val MAE = eval.evaluate(result)
    println(MAE)

   // val RMSE = new RegressionMetrics(predictions.zip(labels)).rootMeanSquaredError
   // val MAE = new RegressionMetrics(predictions.zip(labels)).meanAbsoluteError
    //println("Root mean squared error - " + RMSE)
    //println("Mean absolute error - " + MAE)

    val outresult = spark.createDataFrame(Seq(
      (0L, "Root Mean Squared Error", RMSE.toString),
      (1L,"Mean Absolute Error", MAE.toString)
    ))
    saveToFile(outresult, "regress-result", outputDir)

  }

  /*
    Save dataframe to file
   */

  def saveToFile(dataframe: DataFrame, fileName: String, dir: String): Unit = {
    dataframe.coalesce(1).write.option("header", "true").csv(dir + fileName)
  }


}
