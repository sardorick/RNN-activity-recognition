import pandas as pd 
import numpy as np 


data_X_path  = "C:/Users/Omistaja/Desktop/epicode/rnn-activity-recognition/Train/X_train.txt"
data_y_path  = "C:/Users/Omistaja/Desktop/epicode/rnn-activity-recognition/Train/y_train.txt"
feature_info = "C:/Users/Omistaja/Desktop/epicode/rnn-activity-recognition/features.txt"

data_X = pd.read_csv(data_X_path, delim_whitespace=' ', header=None)
data_y = pd.read_csv(data_y_path,  header=None)
feature_info_names = pd.read_csv(feature_info, header=None)

feature_list = []
for i in feature_info_names[0]:
     feature_list.append(i.strip())
# print(data.head(5))
# data_X
# feature_info_names

selected_feature = []

# feature_columns = feature_list 
data_X.columns = feature_list
# print(feature_list)
for i in  data_X.columns:
     if i.endswith('Mean-1'):
          # print(i)
          selected_feature.append(i)

# print(selected_feature)

selected_X = data_X[selected_feature]
print(selected_X)














































'tBodyAcc-Mean-1'
'tGravityAcc-Mean-1'
'tBodyAccJerk-Mean-1'
'tBodyGyro-Mean-1'
'tBodyGyroJerk-Mean-1'
'tBodyAccMag-Mean-1'
'tGravityAccMag-Mean-1'
'tBodyAccJerkMag-Mean-1'
'tBodyGyroMag-Mean-1'
'tBodyGyroJerkMag-Mean-1'
'fBodyAcc-XYZ-Mean-1'
'fBodyAccJerk-XYZ-Mean-1'
'fBodyGyro-XYZ-Mean-1'
'fBodyAccMag-Mean-1'
'fBodyAccJerkMag-Mean-1'
'fBodyGyroMag-Mean-1'
'fBodyGyroJerkMag-Mean-1'
