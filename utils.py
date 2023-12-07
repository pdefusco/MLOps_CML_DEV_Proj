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