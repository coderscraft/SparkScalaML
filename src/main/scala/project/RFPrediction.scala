package project

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{RandomForestRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, LongType, StructField, StructType}

/**
  * Created by ravirane on 12/11/18.
  */
object RFPrediction {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val trainFile = args(0)
    val predFile = args(1)
    val outPath = args(2)

    // Creating spark session
    val spark = SparkSession.builder()
      .config("spark.master", "local[2]")
      .appName("Zillow Prediction")
      .getOrCreate()

    // Reading Rating and movie files in Dataframe
    var df_train = spark.read.format("com.databricks.spark.csv")
      .option("header", true)
      .schema(customSchema)
      .load(trainFile)

    // Reading Rating and movie files in Dataframe
    var df_predict = spark.read.format("com.databricks.spark.csv")
      .option("header", true)
      .schema(predSchema)
      .load(predFile)

    df_train = df_train.na.drop()
    df_predict = df_predict.na.drop()

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
    val rf = new RandomForestRegressor()
      .setLabelCol("logerror")
      .setFeaturesCol("features")

    /*
      Create pipeline from Assembler and Linear Regression
    */
    val pipeline = new Pipeline()
      .setStages(Array(assembler, rf))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(df_train)

    var fullPredictions = model.transform(df_predict)

    fullPredictions = fullPredictions.select("id_parcel","prediction","prediction","prediction","prediction","prediction","prediction")
      .toDF("ParcelId","201610","201611","201612","201710","201711","201712")

    //fullPredictions.show(200)
    fullPredictions.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true")
      .save(outPath)

  }

  // Schema for Rating file
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

  val predSchema = StructType(Array(
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
    StructField("build_year",LongType,true)))


}
