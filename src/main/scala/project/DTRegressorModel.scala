package project

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel

/**
  * Created by ravirane on 12/11/18.
  * Decision Tree regression model
  */
object DTRegressorModel {

  // Schema for training data
  val customSchema = StructType(Array(
    StructField("id_parcel",LongType,true),
    StructField("tax_year",LongType,true),
    StructField("num_bathroom",LongType,true),
    StructField("num_bedroom",LongType,true),
    StructField("quality",LongType,true),
    StructField("num_bathroom_calc",LongType,true),
    StructField("area_total_calc",LongType,true),
    StructField("census",LongType,true),
    StructField("area_live_finished",LongType,true),
    StructField("no_fip",LongType,true),
    StructField("num_full_bath",LongType,true),
    StructField("heating",LongType,true),
    StructField("tax_land",DoubleType,true),
    StructField("latitude",LongType,true),
    StructField("longitude",LongType,true),
    StructField("area_lot",LongType,true),
    StructField("zoning_landuse",LongType,true),
    StructField("rawcensus",DoubleType,true),
    StructField("region_city",LongType,true),
    StructField("region_county",LongType,true),
    StructField("region_zip",LongType,true),
    StructField("tax_building",LongType,true),
    StructField("tax_property",DoubleType,true),
    StructField("tax_total",LongType,true),
    StructField("num_unit",LongType,true),
    StructField("build_year",LongType,true),
    StructField("logerror",DoubleType,true)))

  def main(args: Array[String]): Unit = {

    //Suppress log to error
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)


    // Creating spark session
    val spark = SparkSession.builder()
      .config("spark.master", "local[2]")
      .appName("Zillow Recomendation model - DecissionTreeRegression")
      .getOrCreate()


    import spark.implicits._

    // Reading Training data from csv
    var df = spark.read.format("com.databricks.spark.csv")
      .option("header", true)
      .schema(customSchema)
      .load("/Users/ravirane/hadoop/project/data/final/final_train.csv")

    //Remove rows with null data
    println("Before" + df.count())
    df = df.na.drop()
    println("After" + df.count())
    df.show()


    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    /*
   Create array of feature columns for regression model
 */
    val featureCols = Array("tax_year","num_bathroom","num_bedroom","quality","num_bathroom_calc","area_total_calc",
      "census","area_live_finished","no_fip","num_full_bath","heating","tax_land","latitude","longitude","area_lot",
      "zoning_landuse","rawcensus","region_city","region_county","region_zip","tax_building","tax_property","tax_total",
      "num_unit","build_year")
    /*
       Create assembler from feature column list
     */
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("logerror")
      .setFeaturesCol("features")

    /*
      Create pipeline from Assembler and Linear Regression
    */
    val pipeline = new Pipeline()
      .setStages(Array(assembler, dt))

    /*
     Create parameter list for cross validation
     - Regulrization maxDepth - 5, 8, 10
   */
    val dtParamMaps = new ParamGridBuilder()
      .addGrid(dt.maxDepth, Array(5, 8, 10))
      .build()

    // rmse evaluation
    var eval = new RegressionEvaluator()
      .setLabelCol("logerror")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    /*
   Build cross validation object with Linear Regression setup
 */
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(dtParamMaps)
      .setEvaluator(eval)
      .setNumFolds(5)

    /*
     Build linear regression model for Training data after cross validation using best outcome
    */

    val cvModel = cv.fit(trainingData)

    /*
   Verify model on test data and generate prediction dataframe
 */
    val fullPredictions = cvModel.transform(testData)

    // Select example rows to display.
    fullPredictions.select("prediction", "logerror", "features").show(15)

    val RMSE = eval.evaluate(fullPredictions)
    /*
    Get best model to findout actual tree generated from it
   */
    val bestModel = cvModel.bestModel

    val treeModel = bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(1).asInstanceOf[DecisionTreeRegressionModel]
    println("Learned classification tree model:\n" + treeModel.toString())

    println("Root Mean Squared Error (RMSE) on test data = " + RMSE)



  }

}
