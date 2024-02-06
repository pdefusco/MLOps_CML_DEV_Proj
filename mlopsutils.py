"""
MLOps Utils
"""


def getModelFromRegistry(experimentName, spark):
    """
    Method to pull Model Artifacts from MLFlow Registry
    Artifacts for the latest MLFlow Tracking Run are obtained only
    """
    
    #Read Experiment ID by Experiment Name
    experiment_id = mlflow.get_experiment_by_name(experimentName).experiment_id
    runs_df = mlflow.search_runs(experiment_id, run_view_type=1)
    print(runs_df['artifact_uri'][0])
    mPath = runs_df['artifact_uri'][0]
    mPath = mPath + "/best-model/sparkml"
    loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=mPath)
    #model = spark.load
    from pyspark.ml.pipeline import PipelineModel
    persistedModel = PipelineModel.load(mPath)
    
    return loaded_model


def defineColTypes(df):
    """
    Method to assign df columns into two lists based on type
    Types are either string or not string
    Input: Spark DF
    Output: Lists by Column Type
    """

    numeric = []
    cat = []

    for col in df.dtypes:
        if col[1] == "string":
            cat.append(col[0])
        else:
            numeric.append(col[0])
    return numeric, cat

def convertToFloat(df, numeric):
    """
    Method to convert Spark DF columns to float
    Does not require input DF columns to be of numeric type only
    Requires input list of numerical columns
    Outputs df with columns in Float format
    """

    for c in df[numeric].columns:
        df = df.withColumn(c, df[c].cast("float"))

    return df