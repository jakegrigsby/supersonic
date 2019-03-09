"""
paramsearch.py 

    1. Generate list of hyperparameters
    2. Generate list of env's to test those params on.
    3. Create list of Task objects using that info.
    4. TrainingManager(task_list, SpeedLearningAgent)
    5. TrainingManager.run(million frames worth of epochs)
    6. TrainingManager.gather_logs
    7. for log in logs: log.make_dataframe() 
    8. Evaluate log dataframes.
    9. Go back to 1.
"""