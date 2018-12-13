/**
  * Created by ravirane on 10/9/18.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession


/*
    Arguments for Running this code - 3 Input path (Path for Ratings.csv, Path for Movies.csv, Output Directory)
    e.g /Users/ravirane/hadoop/Assignment2/data/tmp_rating.csv /Users/ravirane/hadoop/Assignment2/data/movies.csv /Users/ravirane/hadoop/Assignment2/data/output/

    Command to run the MoviePairs -

    time spark-submit --class MoviePairs --master yarn-client --num-executors 1 --driver-memory 512m --executor-memory 512m --executor-cores 1
    ./sparkcs657.jar /user/rrane/data/ratings.csv /user/rrane/data/movies.csv /user/rrane/data/output/
 */

object MoviePairs {


  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      //.config("spark.master", "local[2]")
      .appName("Spark Movie Pairs")
      .getOrCreate()

    // Reading Three input from program arguments
    val ratingFile = args(0)  //Path for Rating file
    val movieFile = args(1)   //Path for Movie file
    var outputFile = args(2)  //Path for output file

    //Below code generate name of output directory to avoid conflict when ran for different input
    val splis = ratingFile.split("/")
    val filename = splis(splis.size -1)
    outputFile = outputFile + "/" + "pairs-" + filename

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
    //Below line generate combinations of all Movie pairs for each user, the group based on Movie pairs to get frequency, and sort based on decending frequency
    // e.g [(MovieID1, MovieId2, Frequency),...]
    val movie_pairs = group_by_user.flatMap(user => user._2.combinations(2)).map(a => ((a(0),a(1)),1)).reduceByKey((a, b) => a+b ).sortBy(_._2, false).map(a => (a._1._1.toInt,a._1._2.toInt,a._2))
    // Generate Movie map from Movie RDD e.g [MovieId->MovieName,..]
    val movieMap = movies_rdd.map(m => (m(0).toString.toInt, m(1).toString)).collectAsMap()
    // Repalce Movie Id with Movie name in initial pairs
    val finalMovies = movie_pairs.map(m => (movieMap.get(m._1), movieMap.get(m._2), m._3))
    // Save final result-Final output look like this - [(Movie1,Movie2,Frequenct),...]
    movie_pairs.saveAsTextFile(outputFile)
  }

}
