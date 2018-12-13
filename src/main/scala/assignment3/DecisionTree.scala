package assignment3

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.tuning.CrossValidator


/**
  * Created by ravirane on 10/28/18.
  *
  * This is decision Tree model which will work on preprocess NY Taxi data and create 6 different dataframe classiefied using
  * Tree model.
  * Trip Time Bin :
  *  0- 5 min => 1, 0 - 10 min => 2  10 - 15 min => 3 15 - 20 min => 4  20 - 30 min => 5 else 6
  *  Command to run job :
  *  time spark-submit --class assignment3.DecisionTree --master yarn-client --num-executors 1 --driver-memory 512m
  *  --executor-memory 512m
  *  --executor-cores 1 ./sparkcs657.jar /user/rrane/data/preprocess.csv /user/rrane/data/output/
  */

object DecisionTree {

  def main(args: Array[String]): Unit = {

    /*
      Object structure to hold the input preprocessed file
     */
    val customSchema = StructType(Array(
      StructField("passenger_count", IntegerType, true),
      StructField("trip_duration", IntegerType, true),
      StructField("day_of_week", IntegerType, true),
      StructField("timeOfDay", IntegerType, true),
      StructField("dist_in_km", IntegerType, true),
      StructField("rain_stat", IntegerType, true),
      StructField("snow_stat", IntegerType, true),
      StructField("duration_bin", IntegerType, true)))

    /*
       Creating Spark instantace to perform further operations
     */
    val spark = SparkSession.builder()
      //.config("spark.master", "local[2]")
      .appName("NY Taxi Decision Tree")
      .getOrCreate()

    /*
       Path for input file and output diretcory
     */
    val trainFile = args(0)
    var outputDir = args(1)

    val train_df = spark.read.format("csv").option("header", "true")
      .schema(customSchema)
      .load(trainFile)

    /*
      Entire training data is used for Decision Tree model
     */
    val training = train_df

    /*
        All feature columns for building Decision Tree
     */
    val featureCols = Array("passenger_count", "day_of_week", "timeOfDay", "dist_in_km", "rain_stat", "snow_stat")

    /*
      Assembler to assemble feature columns
     */
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    /*
        Duration Bin column is the categorical field which we will be predicting
        Features are the remaining column contributing in building tree
     */
    val dTree = new DecisionTreeClassifier().setLabelCol("duration_bin")
      .setFeaturesCol("features")

    /*
       Built pipeline executor from Assembler and Dtree
     */
    val pipeline = new Pipeline()
      .setStages(Array(assembler, dTree))
    // Search through decision tree's maxDepth parameter for best model
    /*
        Param Grid is used for crossvalidation, and attribute we will use is Tree Depth with sizes 2 to 6
     */
    val paramGrid = new ParamGridBuilder().addGrid(dTree.maxDepth, Array(2, 3, 4, 5, 6)).build()

    /*
       Evaluator for evaluating performance factors of outcome model
     */
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("duration_bin")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    /*
      Creating Crossvalidation instance using Evaluator and pipeline. Will perform 10 fold cross validation
     */
    val crossval = new CrossValidator().setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid).setNumFolds(10)

    /*
        Fit crossvalidation model on traing data
     */
    training.show()
    val cvModel = crossval.fit(training)
    /*
      Get best model to findout actual tree generated from it
     */
    val bestModel = cvModel.bestModel
    val treeModel = bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(1).asInstanceOf[DecisionTreeClassificationModel]
    val modelTree = treeModel.toDebugString
    println("Learned classification tree model:\n" + treeModel.toDebugString)
    /*
       Perform predictions using same traing data - This will create prediction column in existing dataframe
    */
    val predictions = cvModel.transform(train_df)
    /*
        Get the accuracy from Evaluator for prediction
     */
    val accuracy = evaluator.evaluate(predictions)
    //println(accuracy)
    //println(evaluator.explainParams())
    /*
       Get all the necessay columns from dataframe for next processing
     */
    val result = predictions.select("passenger_count","day_of_week","timeOfDay","dist_in_km","rain_stat","snow_stat","trip_duration","duration_bin","prediction")
    /*
       Create separe dataframe for each classified label and save it as a csv and use in next regression model
       There are 6 values in class label which are Bins for trip time
     */
    val out_01 = result.filter(predictions("prediction") === 1.0)
    saveToFile(out_01, "out_01", outputDir)
    val out_02 = result.filter(predictions("prediction") === 2.0)
    saveToFile(out_01, "out_02", outputDir)
    val out_03 = result.filter(predictions("prediction") === 3.0)
    saveToFile(out_01, "out_03", outputDir)
    val out_04 = result.filter(predictions("prediction") === 4.0)
    saveToFile(out_01, "out_04", outputDir)
    val out_05 = result.filter(predictions("prediction") === 5.0)
    saveToFile(out_01, "out_05", outputDir)
    val out_06 = result.filter(predictions("prediction") === 6.0)
    saveToFile(out_01, "out_06", outputDir)
    /*
      Save the model accuracy and decision tree in file
     */
    val outresult = spark.createDataFrame(Seq(
      (0L, accuracy.toString),
      (1L, modelTree.toString)
    ))
    saveToFile(outresult, "result", outputDir)
  }

  /*
   Method to save dataframe passed in as a csv
   */
  def saveToFile(dataframe: DataFrame, fileName: String, dir: String): Unit = {
    dataframe.coalesce(1).write.option("header", "true").csv(dir + fileName)
  }

}
