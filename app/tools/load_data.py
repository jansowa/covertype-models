import pandas as pd
from sklearn.model_selection import train_test_split
from app.file_paths import DATASET_PATH
from typing import Tuple


class LoadData:
    __columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2',
                 'Wilderness_Area_3'] + ['Soil_type_' + str(i) for i in range(40)] + ['Cover_Type']
    __X = None
    __y = None
    __X_train = None
    __X_test = None
    __y_train = None
    __y_test = None

    def __init__(self):
        df = pd.read_csv(DATASET_PATH, header=None)
        df.columns = self.__columns
        self.__X = df.drop(columns='Cover_Type')
        self.__y = df["Cover_Type"]
        self.__X_train, self.__X_test, self.__y_train, self.__y_test =\
            train_test_split(self.__X, self.__y, test_size=0.1, random_state=42, stratify=self.__y)

    def load_X_y(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.__X, self.__y

    def load_X_y_splitted(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.__X_train, self.__X_test, self.__y_train, self.__y_test
