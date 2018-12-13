package assignment4

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StopWordsRemover

/**
  * Created by ravirane on 11/11/18.
  * Command - time spark-submit --class assignment4.TopicEda --master yarn-client --num-executors 5
  * --driver-memory 512m --executor-memory 512m --executor-cores 5 ./sparkcs657.jar /user/rrane/data1/
  *
  * This job will perform all data cleaning activities and generate indexed tweets
  * Input - all Tweet csv files
  * Output - CSV file in format -> [{id, Tweet}]
  */
object TopicEda {

  /*
    Structure of tweet file
   */
  val customSchema = StructType(Array(
    StructField("external_author_id", StringType, true),
    StructField("author", StringType, true),
    StructField("content", StringType, true),
    StructField("region", StringType, true),
    StructField("language", StringType, true),
    StructField("publish_date", StringType, true),
    StructField("harvested_date", StringType, true),
    StructField("following", StringType, true),
    StructField("followers", StringType, true),
    StructField("updates", StringType, true),
    StructField("post_type", StringType, true),
    StructField("account_type", StringType, true),
    StructField("new_june_2018", StringType, true),
    StructField("retweet", StringType, true),
    StructField("account_category", StringType, true)))


  def main(args: Array[String]): Unit = {
    // Create Spark Session
    val spark = SparkSession.builder()
      //.config("spark.master", "local[2]")
      .appName("Topic Modeling Preprocessing")
      .getOrCreate()
    // Reading input directory where we have all csv files
    val inputFile = args(0)
    // Create dataframe from all csv files
    val inputDf = spark.read.format("csv")
      .option("header", "true")
      .schema(customSchema)
      .load(inputFile + "*.csv")
    // will process tweets in English language
    // will remove duplicate tweets
    import spark.implicits._
    val proecessDf = inputDf.filter(inputDf("language") === "English").select("content").distinct().map(a => a.getString(0))
      .withColumn("id", monotonically_increasing_id()).select("id","value")
    // Save the output file for LDA processing
    proecessDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(inputFile + "out")

  }


}
