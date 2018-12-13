package project

import assignment5.RecommendationModel.customSchema
import org.apache.spark.sql.types._
import org.apache.spark.sql.{ Dataset, Row, SparkSession}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.sql
import org.apache.log4j.Logger
import org.apache.log4j.Level


import org.apache.spark.mllib.evaluation.RankingMetrics

import scala.collection.mutable.ListBuffer

/**
  * Created by ravirane on 12/10/18.
  */
object PreProcessing {

  //Schema for train data
  val customSchema = StructType(Array(
    StructField("parcelid", LongType, true),
    StructField("logerror", DoubleType, true),
    StructField("transactiondate", StringType, true)))


  def main(args: Array[String]): Unit = {
  // Suppress log to error level
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)


    // Creating spark session
    val spark = SparkSession.builder()
      .config("spark.master", "local[2]")
      .appName("Zillow Recomendation preprocessing")
      .getOrCreate()


    import spark.implicits._

    // Reading properties details file
    var df = spark.read.format("com.databricks.spark.csv")
      .option("header", true)
      //.schema(customSchema)
      .load("/Users/ravirane/hadoop/project/data/properties_2016.csv")

    // Reading Training data file
    var tdf = spark.read.format("com.databricks.spark.csv")
      .option("header", true)
      .schema(customSchema)
      .load("/Users/ravirane/hadoop/project/data/train_2016.csv")

    // Selecting necessary columns from training file
    val train_2016 = tdf.select("parcelid","logerror")

    // Impute properties data for missing values in all column ->
    // Logic is to select max frequency value
    df = performImput(df)

    //Cast properties file to numeric value
    val prop_2016_df = df.select(col("parcelid").cast(LongType),
      col("assessmentyear").cast(IntegerType),
      col("bathroomcnt").cast(IntegerType),
      col("bedroomcnt").cast(IntegerType),
      col("buildingqualitytypeid").cast(IntegerType),
      col("calculatedbathnbr").cast(IntegerType),
      col("calculatedfinishedsquarefeet").cast(IntegerType),
      col("censustractandblock").cast(LongType),
      col("finishedsquarefeet12").cast(IntegerType),
      col("fips").cast(IntegerType),
      col("fullbathcnt").cast(IntegerType),
      col("heatingorsystemtypeid").cast(IntegerType),
      col("landtaxvaluedollarcnt").cast(DoubleType),
      col("latitude").cast(IntegerType),
      col("longitude").cast(IntegerType),
      col("lotsizesquarefeet").cast(IntegerType),
      //col("propertycountylandusecode").cast(IntegerType),
      col("propertylandusetypeid").cast(IntegerType),
     // col("propertyzoningdesc").cast(IntegerType),
      col("rawcensustractandblock").cast(DoubleType),
      col("regionidcity").cast(IntegerType),
      col("regionidcounty").cast(IntegerType),
      col("regionidzip").cast(IntegerType),
      //col("roomcnt").cast(IntegerType),
      col("structuretaxvaluedollarcnt").cast(IntegerType),
      col("taxamount").cast(DoubleType),
      col("taxvaluedollarcnt").cast(IntegerType),
      col("unitcnt").cast(IntegerType),
      col("yearbuilt").cast(IntegerType))

    prop_2016_df.show(20)

    prop_2016_df.describe().filter($"summary" === "count").show

    prop_2016_df.printSchema()

    // Join properties and train data on parcelId column
    var joinedDF = prop_2016_df.as('a).join(
      train_2016.as('b),
      $"a.parcelid" === $"b.parcelid")

    // Post join select necessary columns for modeling
    joinedDF = joinedDF.select($"a.parcelid",$"a.assessmentyear",$"a.bathroomcnt",$"a.bedroomcnt",$"a.buildingqualitytypeid",
      $"a.calculatedbathnbr",$"a.calculatedfinishedsquarefeet",$"a.censustractandblock",$"a.finishedsquarefeet12",$"a.fips",
      $"a.fullbathcnt",$"a.heatingorsystemtypeid",$"a.landtaxvaluedollarcnt",$"a.latitude",$"a.longitude",$"a.lotsizesquarefeet",
      $"a.propertylandusetypeid",$"a.rawcensustractandblock",$"a.regionidcity",$"a.regionidcounty",$"a.regionidzip",
      $"a.structuretaxvaluedollarcnt",$"a.taxamount",$"a.taxvaluedollarcnt",$"a.unitcnt",$"a.yearbuilt",$"b.logerror")

    joinedDF.show()
    joinedDF.count()

    // Save final test data in new file
    prop_2016_df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true")
      .save("/Users/ravirane/hadoop/project/data/final_all")

    // Save final train data in new file
    joinedDF.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true")
      .save("/Users/ravirane/hadoop/project/data/final_joned")

  }


  // Perform data imputation for missing values in each column. missing value is replaced with max frequency value

  def performImput(data: sql.DataFrame): sql.DataFrame = {
    var df = data
    df.groupBy("heatingorsystemtypeid").count().sort(desc("count")).show(5)
    df = df.na.fill("2", Seq("heatingorsystemtypeid"))
    df.groupBy("heatingorsystemtypeid").count().sort(desc("count")).show(5)

    df.groupBy("buildingqualitytypeid").count().sort(desc("count")).show(5)
    df = df.na.fill("7", Seq("buildingqualitytypeid"))
    df.groupBy("buildingqualitytypeid").count().sort(desc("count")).show(5)

    df.groupBy("unitcnt").count().sort(desc("count")).show(5)
    df = df.na.fill("1", Seq("unitcnt"))
    df.groupBy("unitcnt").count().sort(desc("count")).show(5)

    //Will remove this col
    //   df.groupBy("propertyzoningdesc").count().sort(desc("count")).show(5)
    //   df = df.na.fill("7", Seq("propertyzoningdesc"))
    //   df.groupBy("propertyzoningdesc").count().sort(desc("count")).show(5)

    df.groupBy("lotsizesquarefeet").count().sort(desc("count")).show(5)
    df = df.na.fill("6000.0", Seq("lotsizesquarefeet"))
    df.groupBy("lotsizesquarefeet").count().sort(desc("count")).show(5)

    df.groupBy("finishedsquarefeet12").count().sort(desc("count")).show(5)
    df = df.na.fill("1440", Seq("finishedsquarefeet12"))
    df.groupBy("finishedsquarefeet12").count().sort(desc("count")).show(5)

    df.groupBy("fullbathcnt").count().sort(desc("count")).show(5)
    df = df.na.fill("2", Seq("fullbathcnt"))
    df.groupBy("fullbathcnt").count().sort(desc("count")).show(5)

    df.groupBy("calculatedbathnbr").count().sort(desc("count")).show(10)
    df = df.na.fill("2.0", Seq("calculatedbathnbr"))
    df.groupBy("calculatedbathnbr").count().sort(desc("count")).show(10)

    df.groupBy("censustractandblock").count().sort(desc("count")).show(5)
    df = df.na.fill("60379203391054", Seq("censustractandblock"))
    df.groupBy("censustractandblock").count().sort(desc("count")).show(5)

    df.groupBy("landtaxvaluedollarcnt").count().sort(desc("count")).show(5)
    df = df.na.fill("20980.0", Seq("landtaxvaluedollarcnt"))
    df.groupBy("landtaxvaluedollarcnt").count().sort(desc("count")).show(5)

    df.groupBy("regionidcity").count().sort(desc("count")).show(5)
    df = df.na.fill("12447", Seq("regionidcity"))
    df.groupBy("regionidcity").count().sort(desc("count")).show(5)

    df.groupBy("yearbuilt").count().sort(desc("count")).show(10)
    df = df.na.fill("1955.0", Seq("yearbuilt"))
    df.groupBy("yearbuilt").count().sort(desc("count")).show(10)

    df.groupBy("calculatedfinishedsquarefeet").count().sort(desc("count")).show(5)
    df = df.na.fill("1440.0", Seq("calculatedfinishedsquarefeet"))
    df.groupBy("calculatedfinishedsquarefeet").count().sort(desc("count")).show(5)

    df.groupBy("structuretaxvaluedollarcnt").count().sort(desc("count")).show(10)
    df = df.na.fill("100000.0", Seq("structuretaxvaluedollarcnt"))
    df.groupBy("structuretaxvaluedollarcnt").count().sort(desc("count")).show(10)

    df.groupBy("taxvaluedollarcnt").count().sort(desc("count")).show(5)
    df = df.na.fill("400000.0", Seq("taxvaluedollarcnt"))
    df.groupBy("taxvaluedollarcnt").count().sort(desc("count")).show(5)


    df.groupBy("taxamount").count().sort(desc("count")).show(5)
    df = df.na.fill("345.72", Seq("taxamount"))
    df.groupBy("taxamount").count().sort(desc("count")).show(5)


    df.groupBy("regionidzip").count().sort(desc("count")).show(5)
    df = df.na.fill("96987", Seq("regionidzip"))

   // df.groupBy("propertycountylandusecode").count().sort(desc("count")).show(5)
   // df = df.na.fill("0100", Seq("propertycountylandusecode"))

    //should remove this
    //df.groupBy("roomcnt").count().sort(desc("count")).show(5)
    //df = df.na.fill("0100", Seq("roomcnt"))


    df.groupBy("bathroomcnt").count().sort(desc("count")).show(5)
    df = df.na.fill("2.0", Seq("bathroomcnt"))

    df.groupBy("bedroomcnt").count().sort(desc("count")).show(5)
    df = df.na.fill("3.0", Seq("bedroomcnt"))

    df.groupBy("assessmentyear").count().sort(desc("count")).show(5)
    df = df.na.fill("2015", Seq("assessmentyear"))
    df.groupBy("assessmentyear").count().sort(desc("count")).show(5)

    df.groupBy("regionidcounty").count().sort(desc("count")).show(5)
    df = df.na.fill("3101", Seq("regionidcounty"))
    df.groupBy("regionidcounty").count().sort(desc("count")).show(5)

    df.groupBy("rawcensustractandblock").count().sort(desc("count")).show(5)
    df = df.na.fill("060590524.262052", Seq("rawcensustractandblock"))


    df.groupBy("propertylandusetypeid").count().sort(desc("count")).show(5)
    df = df.na.fill("261", Seq("propertylandusetypeid"))

    df.groupBy("assessmentyear").count().sort(desc("count")).show(5)
    df = df.na.fill("0100", Seq("assessmentyear"))


    df.groupBy("longitude").count().sort(desc("count")).show(5)
    df = df.na.fill("-118390000", Seq("longitude"))


    df.groupBy("latitude").count().sort(desc("count")).show(5)
    df = df.na.fill("33984600", Seq("latitude"))


    df.groupBy("fips").count().sort(desc("count")).show(5)
    df = df.na.fill("06037", Seq("fips"))
    df
  }


}
