import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector


// get data from hdfs

val tweet = sc.textFile("wasb://bigdata01@bigdatastore01.blob.core.windows.net/user/CentrikAlpha01/tweets_words.csv")
tweet.take(10)

val tweet_test = sc.textFile("wasb://bigdata01@bigdatastore01.blob.core.windows.net/user/CentrikAlpha01/tweets_words_test.csv")
tweet_test.take(10)

// coding  tweets 
val hashingTF = new HashingTF(1000)

val parsedData = tweet.map { line =>
  val parts = line.split(',')
  LabeledPoint(parts(0).toDouble, hashingTF.transform(parts.tail))
}



// Split data into training (60%) and test (40%).
val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0)
val test = splits(1)

val model = NaiveBayes.train(training, lambda = 1.0)
val prediction = model.predict(test.map(_.features))


for (i <- prediction) {
  print(i)
  if (i == 1) println("   Personal Life Event in this tweet is : WEDDING") 
  else if (i == 2 ) println (",   Personal Life Event in this tweet is: GRADUATE")
  else if (i == 3 ) println (",   Personal Life Event in this tweet is : NEW JOB") 
  else if (i == 4 ) println (",   Personal Life Event in this tweet is : BIRTH IN FAMILIE") 
  else if (i == 0 ) println (",   Personal Life Event in this tweet is : OTHER")}

val predictionAndLabel = prediction.zip(test.map(_.label))
val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()






val parsedData_test = tweet_test.map { line =>
  val parts = line.split(',')
  hashingTF.transform(parts)
}

val prediction_test = model.predict(parsedData_test)
prediction_test.take(10)


for (i <- prediction_test) {
  print(i)
  if (i == 1) println("   Personal Life Event in this tweet is : WEDDING") 
  else if (i == 2 ) println (",   Personal Life Event in this tweet is : GRADUATE")
  else if (i == 3 ) println (",   Personal Life Event in this tweet is : NEW JOB") 
  else if (i == 4 ) println (",   Personal Life Event in this tweet is : BIRTH IN FAMILIE") 
  else if (i == 0 ) println (",   Personal Life Event in this tweet is: OTHER")}
