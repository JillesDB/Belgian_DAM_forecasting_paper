import Generate_Evaluate_and_Time_Forecast
import os


cleaned_code_path = os.path.dirname(os.path.abspath(__file__))
print(cleaned_code_path)

# Define relative paths from the 'Cleaned_code' directory
datasets_path = os.path.join(cleaned_code_path, 'Datasets')
forecasts_folder = os.path.join(cleaned_code_path, 'Folder_Parallel_Run_2')
real_prices_path = os.path.join(datasets_path, 'Real_prices.csv')


#Everything to run for the less frequently calibrated models.
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Full_Dataset',path_datasets_folder=datasets_path
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Full_Dataset',path_datasets_folder=datasets_path
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=7,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Full_Dataset',path_datasets_folder=datasets_path
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=28,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Full_Dataset',path_datasets_folder=datasets_path
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=84,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Full_Dataset',path_datasets_folder=datasets_path
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=168,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Full_Dataset',path_datasets_folder=datasets_path
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1093,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)



#Everything to run for the one variable family models.
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='ModelBELoad_dataframe',path_datasets_folder=r'C:\Users\jille\Documents\2e_Master_2022_2023\Masterthesis\Code\Local_repo\Cleaned_code\Datasets\Dataframes_one_ex_var'
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='ModelBenchmark_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_one_ex_var')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='ModelCHPrice_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_one_ex_var')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='ModelFF_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_one_ex_var')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='ModelFR_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_one_ex_var')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='ModelSolar_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_one_ex_var')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='ModelWind_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_one_ex_var')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)

#Everything to run for the several variable family models.
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Model3Vars_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_several_vars')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Model5Vars_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_several_vars')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Model8Vars_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_several_vars')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)
Generate_Evaluate_and_Time_Forecast.Predict_Evaluate_and_Time(path_real_prices=real_prices_path
,path_forecasts_folder=forecasts_folder
,name_dataframe='Model10Vars_dataframe',path_datasets_folder=os.path.join(datasets_path, 'Dataframes_several_vars')
,begin_test_date='2020-01-01', end_test_date='2022-12-31',recalibration_window=1,calibration_window_set={56,84,112,714,721,728},
regular=1,weighed=1)






