import org.apache.spark.sql.SparkSession

/**
  * Created by ravirane on 10/10/18.
  */

/*
    Arguments for Running this code - 3 Input path (Path for Ratings.csv, Path for Movies.csv, Output Directory)
    e.g /Users/ravirane/hadoop/Assignment2/data/tmp_rating.csv /Users/ravirane/hadoop/Assignment2/data/movies.csv /Users/ravirane/hadoop/Assignment2/data/output/

    Command to run the MoviePairs -

    time spark-submit --class Lift --master yarn-client --num-executors 1 --driver-memory 512m --executor-memory 512m --executor-cores 1
    ./sparkcs657.jar /user/rrane/data/ratings.csv /user/rrane/data/movies.csv /user/rrane/data/output/

    This file is samilar to MovieStripe, Only new code is in getting Lift, hence comments acre added in that section only
 */


object Lift {
  type stripeCollector = (Int, scala.collection.mutable.Map[Int, Int])

  def getPattern(a: List[Int]): List[stripeCollector] = {
    var x = List[stripeCollector]()
    for(i <- a.indices) {
      var dict = scala.collection.mutable.Map[Int, Int]()
      for(j <- a.indices) {
        if(j > i) {
          dict += (a(j) -> 1)
        }
      }
      if(i < a.size - 1) {
        x = x:+ (a(i), dict)
      }
    }
    x
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      //.config("spark.master", "local[2]")
      .appName("Spark Lift")
      .getOrCreate()

    val ratingFile = args(0)
    val movieFile = args(1)
    var outputFile = args(2)

    val splis = ratingFile.split("/")
    val filename = splis(splis.size -1)
    outputFile = outputFile + "/" + "Lift-" + filename

    val rating_df = spark.read.format("csv").option("header", "false").load(ratingFile)
    val movies_df = spark.read.format("csv").option("header", "false").load(movieFile)
    val rating_rdd = rating_df.rdd
    val movies_rdd = movies_df.rdd

    val highRatings = rating_rdd.filter(row => row(2).toString.toDouble >= 4.0).map(row => (row(0).toString.toInt,row(1).toString.toInt))

    val clean_movies_rdd = movies_rdd.map(m => (m(0).toString.toInt, m(1).toString))

    val group_by_user = highRatings.combineByKey((x:Int) => List(x), (acc:List[Int], x) => acc:+x , (acc1:List[Int], acc2:List[Int]) => acc1 ::: acc2)
    val movie_dict = group_by_user.flatMap(user => getPattern(user._2))
    val createCombiner = (stripeMovie: scala.collection.mutable.Map[Int, Int]) => stripeMovie
    val combineStripe = (collector: scala.collection.mutable.Map[Int, Int], stripeMovie: scala.collection.mutable.Map[Int, Int]) => {
      stripeMovie.foreach(keyVal => {
        if(collector.contains(keyVal._1.toInt)) {
          val mapVal = collector.getOrElse(keyVal._1, 0)
          collector += keyVal._1.toInt -> (mapVal + keyVal._2)
        } else {
          collector += keyVal._1.toInt -> keyVal._2
        }
      })
      collector
    }

    val mergeStripe = (collector1: scala.collection.mutable.Map[Int, Int], collector2: scala.collection.mutable.Map[Int, Int]) => {
      collector1.foreach(keyVal => {
        if(collector2.contains(keyVal._1.toInt)) {
          val mapVal = collector2.getOrElse(keyVal._1, 0)
          collector1 += keyVal._1.toInt -> (keyVal._2 + mapVal)
        } else {
          collector1 += keyVal._1.toInt -> keyVal._2
        }
      })
      collector1
    }


    val finalMovieDict = movie_dict.combineByKey(createCombiner, combineStripe, mergeStripe)

    // Above this line all code same as MovieStripe file and same cooments are applicable

    //This line generate Map for all movies and their frequencies
    // e.g   MovieId1 -> 10, MovieId2 -> 30
    val movieFreqMap = highRatings.map(a => (a._2, 1)).reduceByKey((a, b) => a + b).sortBy(_._2, false).collectAsMap()
    //Getting the total rating record count
    val recordSize = highRatings.count()
    // Below line has two loop for generating movie pair from Stripes, and finding their Lift and returning only
    // those pairs have Lift value more than 2.5
    // Outcome of this block is [(movieIdA, MovieIdB, Lift),...]
    val innerStripe = finalMovieDict.flatMap(m => {
      // Frequency of key - Movie A
      val freqA = movieFreqMap.getOrElse(m._1,1).toFloat
      // Finding probability of Movie A
      val pA = freqA/recordSize
      // Accumulator for creating all pairs of Movie A having Lift > 2.5
      var x = List[(Int,Int, Double)]()
      m._2.foreach(keyVal => {
        // Get frequency of Movie B
        val freqB = movieFreqMap.getOrElse(keyVal._1,1).toFloat
        // Find probability of Movie B
        val pB = freqB/recordSize
        // Find combine probability of Movie A and B
        val pAB = keyVal._2.toFloat/recordSize
        // Find Lift = p(AB)/(p(A)*p(B))
        val lift = pAB/(pA*pB)
        // If lift is > 2.5 the add in return list
        if(lift > 2.5) {
          x = x :+ (m._1, keyVal._1,lift.toDouble)
         }
      })
      //Return list will be flatten by flatMap
      x
    })
    // Generate Movie map from Movie RDD e.g [MovieId->MovieName,..]
    val movieMap = movies_rdd.map(m => (m(0).toString.toInt, m(1).toString)).collectAsMap()
    // Repalce Movie Id with Movie name in initial pairs
    val finalStripe = innerStripe.map(m => (movieMap.getOrElse(m._1,""),movieMap.getOrElse(m._2,""),m._3))

    //finalStripe.sortBy(_._3, false).foreach(println)
    // Save final result-Final output look like this - [(Movie1,Movie2,Frequenct),...]
    finalStripe.sortBy(_._3, false).saveAsTextFile(outputFile)
  }
}
