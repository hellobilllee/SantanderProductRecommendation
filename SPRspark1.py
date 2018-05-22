from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import explode
import itertools
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
import pandas as pd
import datetime
import gzip

conf = SparkConf()
conf.set("spark.executor.memory", "6G")
conf.set("spark.driver.memory", "2G")
conf.set("spark.executor.cores", "4")

conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.default.parallelism", "4")
conf.setMaster('local[4]')

spark = SparkSession \
    .builder.config(conf=conf) \
    .appName("spark-santander-recommendation").getOrCreate()

selectd_features = [
    'ncodpers','fecha_dato',
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
    'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
    'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
used_features = list(set(selectd_features)-set("fecha_dato"))
# spark is an existing SparkSession
full = spark.read.csv("train_ver2.csv", header="true",inferSchema="true")\
    .select(*selectd_features)
df_train = full[(full["fecha_dato"]=="2015-06-28")].select(*used_features)
df_val = full[(full["fecha_dato"]=="2016-06-28")].select(*used_features)
print(df_train.describe())

lista_users= pd.read_csv("test_ver2.csv", usecols=["ncodpers"])
lista_products=['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
item_cols = [i for i in range(24)]

ratingsRDD = df_train.rdd.map(lambda p: Row(userId=p[0], itemCol=item_cols,rating=p[-24:]))
ratings_train = spark.createDataFrame(ratingsRDD)
ratingsRDD_val = df_val.rdd.map(lambda p: Row(userId=p[0], itemCol=item_cols,rating=p[-24:]))
ratings_val = spark.createDataFrame(ratingsRDD_val)

ratings=ratings_train.select( 'userId','itemCol','rating')
val=ratings_val.select( 'userId','itemCol','rating')

ratings=ratings.select('userId', explode('itemCol'),'rating')
val=val.select('userId', explode('itemCol'),'rating')
ratingsRDD = ratings.rdd.map(lambda p: Row(userId=p[0],itemCol=p[1],ranking=p[2][int(p[1])]))
ratings2 = ratingsRDD.toDF()
ratingsRDD_val = val.rdd.map(lambda p: Row(userId=p[0],itemCol=p[1],ranking=p[2][int(p[1])]))
validation = ratingsRDD_val.toDF()
training=ratings2.withColumn("userId", ratings2["userId"].cast("int")).withColumn("itemCol", ratings2["itemCol"].cast("int")).withColumn("ranking", ratings2["ranking"].cast("int"))
test=validation.withColumn("userId", validation["userId"].cast("int")).withColumn("itemCol", validation["itemCol"].cast("int")).withColumn("ranking", validation["ranking"].cast("int"))
training=training.na.fill(0)
train, val=training.randomSplit([0.8,0.2],10)

sc=spark.sparkContext
sc.setCheckpointDir('checkpoint/')

evaluator = RegressionEvaluator(metricName="rmse", labelCol="ranking",
                                predictionCol="prediction")
def computeRmse(model, data):
    """
    Compute RMSE (Root mean Squared Error).
    """
    predictions = model.transform(data)
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))
    return rmse

#train models and evaluate them on the validation set

ranks = [15]
lambdas = [0.05]
numIters = [30]
bestModel = None
bestValidationRmse = float("inf")
bestRank = 0
bestLambda = -1.0
bestNumIter = -1
training=training.na.drop()
test=test.na.drop()
val=val.na.drop()
for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
    als = ALS(rank=rank, maxIter=numIter, regParam=lmbda, numUserBlocks=10, numItemBlocks=10, implicitPrefs=False,
              alpha=1.0,
              userCol="userId", itemCol="itemCol", seed=1, ratingCol="ranking", nonnegative=True,
              checkpointInterval=10, intermediateStorageLevel="MEMORY_AND_DISK", finalStorageLevel="MEMORY_AND_DISK")
    model=als.fit(training)

    validationRmse = computeRmse(model, val)
    print ("RMSE (validation) = %f for the model trained with " % validationRmse + \
            "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, numIter))
    if (validationRmse< bestValidationRmse):
        bestModel = model
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lmbda
        bestNumIter = numIter

model=bestModel

model_path = "model/"
model.save(model_path)

topredict=test[test['ranking']==0]
predictions=model.transform(topredict)

Recom=predictions.rdd.map(lambda p: Row(user=p[2],ProductPredictions=(p[0],p[3]))).toDF()
ppT=Recom.groupby("user").agg(F.collect_list("ProductPredictions"))
ppT=ppT.withColumn("ncodpers", ppT["user"].cast("int")).withColumn("itemCol", ppT["collect_list(ProductPredictions)"]).drop('user',"collect_list(ProductPredictions)")
ppTP=ppT.toPandas()
print(ppTP.head(10))
predFin=ppTP.merge(lista_users,how="right",on='ncodpers')

def create_submission(preds, target_cols):
    print('Saving results on disk')
    info_string = 'ALS'
    now = datetime.datetime.now()
    sub_file = 'ppT-submission_' + info_string + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    # Create the submission text
    print('Creating text...')
    text = 'ncodpers,added_products\n'
    for i, ncodpers in enumerate(preds.ncodpers):
        text += '%i,' % ncodpers
        item = predFin["itemCol"].values[i]
        # print item
        newitem = sorted(item, key=lambda x: x[1], reverse=True)

        for j in range(len(newitem)):
            text += '%s ' % lista_products[newitem[j][0]]

        text += '\n'
    # Write to file
    print("writing to file")
    with gzip.open('%s.gz' % sub_file, 'w') as f:
        f.write(text)

create_submission(predFin, lista_products)






