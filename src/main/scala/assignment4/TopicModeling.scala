package assignment4

import scala.collection.mutable
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType,LongType, StringType, StructField, StructType}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.clustering.{DistributedLDAModel, LDA}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.StopWordsRemover
import scala.collection.mutable.ListBuffer
import org.apache.spark.sql.Column


/**
  * Created by ravirane on 11/11/18.
  */
object TopicModeling {

  // Structure of input file
  val customSchema = StructType(Array(
    StructField("id", LongType, true),
    StructField("value", StringType, true)))


  def main(args: Array[String]): Unit = {
    /*
       Create spark session
    */
    val spark = SparkSession.builder()
      //.config("spark.master", "local[2]")
      .appName("Topic Modeling")
      .getOrCreate()
    // Reading input file and output directory
    val inputPath = args(0)

    /*
       Read data file and create dataframe
     */
    val inputDf = spark.read.format("csv").option("header", "true")
      .schema(customSchema)
      .load(inputPath + "*.csv")
    /**
      * Below processing will transform [{id, tweet}] file to [{id, tweet, [words]}] form and construct vocab
      * 1. Will tokenize words
      * 2. Take workd length > 4
      * 3. Word should only be in English
      */
    import spark.implicits._
    var step1DF = inputDf.map(a => ( a.getLong(0), a.getString(1), a.getString(1).toLowerCase.split("\\s")))
      .map(a => (a._1, a._2, a._3.filter(_.length > 4).filter(_.forall(java.lang.Character.isLetter))))
      .toDF("id", "tweet", "raw")

    /**
      * There is a default list of stopword in StopWordsRemover package
      * Will remove those words as well and transform dataframe
      */
    val remover = new StopWordsRemover()
      .setInputCol("raw")
      .setOutputCol("words")

    val processedDF = remover.transform(step1DF).select("id","tweet","words")

    //Select word RDD
    val wordsCol = processedDF.select("words").map(a => a.getSeq[String](0)).rdd

    // Choose the vocabulary
    // termCounts: Sorted list of (term, termCount) pairs
    val termCounts: Array[(String, Long)] = wordsCol.flatMap(_.map(_ -> 1L))
      .reduceByKey(_+_).collect().sortBy(-_._2)
    // Store most frequent words
    var allTimeFrequent = new ListBuffer[(String,Long)]()
    termCounts.take(100).foreach(a => allTimeFrequent += a)
    allTimeFrequent.toDF().coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(inputPath + "allTimeFreq")

    // Will remove most common 25 terms from vocab
    val numStopwords = 25
    val vocabArray: Array[String] = termCounts.takeRight(termCounts.size - numStopwords).map(_._1)
    // Generate Index for Vocab
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap
    // Genearate RDD from processed DF
    val tokenized = processedDF.map(a => (a.getLong(0), a.getString(1), a.getSeq[String](2))).rdd
   // Build documents from Vocab and Original tweet words.
    // Document Structure is {id, [vocabSize,[[wordIndx],[Count]]]}
    //This document is then passed to LDA for topic generation
    val documents: RDD[(Long, Vector)] =
      tokenized.map { case (id, tweet, words) =>
        val counts = new mutable.HashMap[Int, Double]()
        words.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }
    //Perform experiment for different topic size, now for 5
    val numTopics = 5
    // Set iteration to some no, now 20
    val lda = new LDA().setK(numTopics).setMaxIterations(20)
    // Run LDA model on documents
    val ldaModel = lda.run(documents)
    // Generate and save top 100 terms
    var allTopicsTerm = new ListBuffer[(String,Double)]()
    termCounts.take(100).foreach(a => allTimeFrequent += a)

    // Save topics, showing top-weighted 30 terms for each topic.
    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 30)
    var i = 0
    topicIndices.foreach { case (terms, termWeights) =>
      allTopicsTerm.append(("Topic",1.0 + i))
      i = i + 1
      terms.zip(termWeights).foreach { case (term, weight) =>
        allTopicsTerm.append((vocabArray(term.toInt),weight))
      }
    }
    allTopicsTerm.toDF().coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(inputPath + "allTopicsTerm")


   // Save topics and 20 tweets in each topic
    var distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]

    val topDocPerTopic = new ListBuffer[(String,Double)]()
    var docspertopic = distLDAModel.topDocumentsPerTopic(maxDocumentsPerTopic = 20)
    var j = 0
    docspertopic.foreach { case (terms, termWeights) =>
      topDocPerTopic.append(("Topic",1.0 + j))
      j = j + 1
      terms.zip(termWeights).foreach { case (id, weight) =>
        val doc = processedDF.filter($"id" === id).select("tweet").take(1)
        topDocPerTopic.append((doc(0).getString(0), weight))
      }
    }
    topDocPerTopic.toDF().coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save(inputPath + "topDocPerTopic")

  }




}
