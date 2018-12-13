import org.apache.spark.sql.SparkSession

/**
  * Created by ravirane on 10/9/18.
  */

/*
    Arguments for Running this code - 3 Input path (Path for Ratings.csv, Path for Movies.csv, Output Directory)
    e.g /Users/ravirane/hadoop/Assignment2/data/tmp_rating.csv /Users/ravirane/hadoop/Assignment2/data/movies.csv /Users/ravirane/hadoop/Assignment2/data/output/

    Command to run the MoviePairs -

    time spark-submit --class MovieStripe --master yarn-client --num-executors 1 --driver-memory 512m --executor-memory 512m --executor-cores 1
    ./sparkcs657.jar /user/rrane/data/ratings.csv /user/rrane/data/movies.csv /user/rrane/data/output/
 */

object MovieStripe {

  type stripeCollector = (Int, scala.collection.mutable.Map[Int, Int])

  /*
    getPattern(List) method generate stripe pattern. Input is a list of numbers for which we want all stripe
    e.g input = [1,2,3,4]
        output = [
              (1,(2->1,3->1,4->1)),
              (2,(3->1,4->1)),
              (3,(4->))
        ]

        ->1 is initial count for each associated movie
   */
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
      .appName("Spark Movie Stripe")
      .getOrCreate()

    // Reading Three input from program arguments
    val ratingFile = args(0) //Path for Rating file
    val movieFile = args(1)  //Path for Movie file
    var outputFile = args(2) //Path for output file

    //Below code generate name of output directory to avoid conflict when ran for different input
    val splis = ratingFile.split("/")
    val filename = splis(splis.size -1)
    outputFile = outputFile + "/" + "stripe-" + filename

    // Reading Rating and Movies csv files
    val rating_df = spark.read.format("csv").option("header", "false").load(ratingFile)
    val movies_df = spark.read.format("csv").option("header", "false").load(movieFile)

    // Changing Ratings and Movies dataframe to RDD for processing
    val rating_rdd = rating_df.rdd
    val movies_rdd = movies_df.rdd

    //This line filter RDD for high rating movies, keep only UserId and MovieId and create PairRDD
    val highRatings = rating_rdd.filter(row => row(2).toString.toDouble >= 4.0).map(row => (row(0).toString.toInt,row(1).toString.toInt))
    //PairRDD is grouped based on UserId and generate pairRDD for userId and List Of High rated Movies e.g 12 -> [1, 4, 7], 20 -> [1, 6, 9, 12]
    val group_by_user = highRatings.combineByKey((x:Int) => List(x), (acc:List[Int], x) => acc:+x , (acc1:List[Int], acc2:List[Int]) => acc1 ::: acc2)
    //using getPattern method, stripes are generated for each user
    val movie_dict = group_by_user.flatMap(user => getPattern(user._2))
    // Initialize combiner for merging stripes across different users
    val createCombiner = (stripeMovie: scala.collection.mutable.Map[Int, Int]) => stripeMovie
    // Combine stripes in same worker logic
    // e.g (1,(2->1, 3->1, 4->1)) + (1,(2->1)) => (1,(2->2, 3->1, 4->1))
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
   // Same combiner logic but across workers
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

    // Combine stripes from all userIds using above three function to generate actual stripe for MoviesId's
    val finalMovieDict = movie_dict.combineByKey(createCombiner, combineStripe, mergeStripe)
    // Generate Movie map from Movie RDD e.g [MovieId->MovieName,..]
    val movieMap = movies_rdd.map(m => (m(0).toString.toInt, m(1).toString)).collectAsMap()
    // Repalce Movie Id for Each stripe Key and All stripe values
    val finalStripe = finalMovieDict.map(m => {
      var dict = scala.collection.mutable.Map[String, Int]()
      m._2.foreach(keyVal => {
        dict += movieMap.getOrElse(keyVal._1,"") -> keyVal._2
      })
      (movieMap.getOrElse(m._1,""), dict)
    })
    // Save final result-Final output look like this - [(Movie1(Movie2->10,Movie3->5..),...]
    finalStripe.saveAsTextFile(outputFile)
  }

}
