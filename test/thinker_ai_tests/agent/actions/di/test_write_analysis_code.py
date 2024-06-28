import asyncio
import json

import pytest

from thinker_ai.agent.actions.di.write_analysis_code import WriteAnalysisCode
from thinker_ai.agent.provider.schema import Message


@pytest.mark.asyncio
async def test_write_code_with_plan():
    write_code = WriteAnalysisCode()

    user_requirement = "Run data analysis on sklearn Iris dataset, include a plot"
    plan_status = "\n## Finished Tasks\n### code\n```python\n\n```\n\n### execution result\n\n\n## Current Task\nLoad the sklearn Iris dataset and perform exploratory data analysis\n\n## Task Guidance\nWrite complete code for 'Current Task'. And avoid duplicating code from 'Finished Tasks', such as repeated import of packages, reading data, etc.\nSpecifically, \nThe current task is about exploratory data analysis, please note the following:\n- Distinguish column types with `select_dtypes` for tailored analysis and visualization, such as correlation.\n- Remember to `import numpy as np` before using Numpy functions.\n\n"

    code = await write_code.run(user_requirement=user_requirement, plan_status=plan_status)
    assert len(code) > 0
    assert "sklearn" in code


@pytest.mark.asyncio
async def test_write_code_with_tools():
    write_code = WriteAnalysisCode()

    user_requirement = "Preprocess sklearn Wine recognition dataset and train a model to predict wine class (20% as validation), and show validation accuracy."
    tool_info = """
    ## Capabilities
    - You MUST utilize pre-defined tools in any code lines from 'Available Tools' in the form of Python class or function.
    - You can freely combine the use of any other public packages, like sklearn, numpy, pandas, etc..

    ## Available Tools:
    Each tool is described in JSON format. when you call a tool, and import tool path first, they are in the file thinker_ai/agent/tools/libs/data_preprocess.py
{
  'FillMissingValue': {
    'type': 'class',
    'description': 'Completing missing values with simple strategies.',
    'methods': {
      '__init__': {
        'type': 'function',
        'description': 'Initialize self. ',
        'signature': '(self, features: \'list\', strategy: "Literal[\'mean\', \'median\', \'most_frequent\', \'constant\']" = \'mean\', fill_value=None)',
        'parameters': 'Args: features (list): Columns to be processed. strategy (Literal["mean", "median", "most_frequent", "constant"], optional): The imputation strategy, notice \'mean\' and \'median\' can only be used for numeric features. Defaults to \'mean\'. fill_value (int, optional): Fill_value is used to replace all occurrences of missing_values. Defaults to None.'
      },
      'fit': {
        'type': 'function',
        'description': 'Fit a model to be used in subsequent transform. ',
        'signature': "(self, df: 'pd.DataFrame')",
        'parameters': 'Args: df (pd.DataFrame): The input DataFrame.'
      },
      'fit_transform': {
        'type': 'function',
        'description': 'Fit and transform the input DataFrame. ',
        'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'",
        'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'
      },
      'transform': {
        'type': 'function',
        'description': 'Transform the input DataFrame with the fitted model. ',
        'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'",
        'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'
      }
    }
  }
"""

    code = await write_code.run(user_requirement=user_requirement, tool_info=tool_info)
    assert len(code) > 0
    assert "thinker_ai.agent.tools.libs" in code


@pytest.mark.asyncio
async def test_debug_with_reflection():
    user_requirement = "read a dataset test.csv and print its head"

    plan_status = """
    ## Finished Tasks
    ### code
    ```python
    ```

    ### execution result

    ## Current Task
    import pandas and load the dataset from 'test.csv'.

    ## Task Guidance
    Write complete code for 'Current Task'. And avoid duplicating code from 'Finished Tasks', such as repeated import of packages, reading data, etc.
    Specifically, 
    """

    wrong_code = """import pandas as pd\ndata = pd.read_excel('test.csv')\ndata"""  # use read_excel to read a csv
    error = """
    Traceback (most recent call last):
        File "<stdin>", line 2, in <module>
        File "/Users/gary/miniconda3/envs/py39_scratch/lib/python3.9/site-packages/pandas/io/excel/_base.py", line 478, in read_excel
            io = ExcelFile(io, storage_options=storage_options, engine=engine)
        File "/Users/gary/miniconda3/envs/py39_scratch/lib/python3.9/site-packages/pandas/io/excel/_base.py", line 1500, in __init__
            raise ValueError(
        ValueError: Excel file format cannot be determined, you must specify an engine manually.
    """
    working_memory = [
        Message(content=wrong_code, role="assistant"),
        Message(content=error, role="user"),
    ]
    new_code = await WriteAnalysisCode().run(
        user_requirement=user_requirement,
        plan_status=plan_status,
        working_memory=working_memory,
        use_reflection=True,
    )
    assert "read_csv" in new_code  # should correct read_excel to read_csv


# @pytest.mark.asyncio
# async def test_run_with_reflection():
#     plan_status = """
#     ## Finished Tasks
#     ### code
#     ```python
#     import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# train_data_path = '../data/titanic/train.csv'
# df = pd.read_csv(train_data_path)
# df.info()
# df.head()
# numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
# print('Numerical columns:', numerical_cols)
# print('Categorical columns:', categorical_cols)
# print(df[numerical_cols].describe())
# print(df[categorical_cols].describe())
# plt.figure(figsize=(10, 6))
# sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
# plt.title('Missing Values Heatmap')
# plt.show()
# plt.figure(figsize=(12, 8))
# corr_matrix = df[numerical_cols].corr()
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Matrix')
# plt.show()
# for col in numerical_cols:
#     plt.figure(figsize=(10, 6))
#     sns.histplot(df[col].dropna(), kde=True)
#     plt.title(f'Distribution of {col}')
#     plt.show()
# for col in categorical_cols:
#     plt.figure(figsize=(10, 6))
#     sns.countplot(x=col, data=df)
#     plt.title(f'Count Plot of {col}')
#     plt.show()
#
# import pandas as pd
# import numpy as np
# from thinker_ai.agent.tools.libs.feature_engineering import TargetMeanEncoder, GeneralSelection
# from thinker_ai.agent.tools.libs.data_preprocess import StandardScale, LabelEncode
# train_data_path = '../data/titanic/train.csv'
# eval_data_path = '../data/titanic/test.csv'
# train_df = pd.read_csv(train_data_path)
# eval_df = pd.read_csv(eval_data_path)
# train_df_copy = train_df.copy()
# eval_df_copy = eval_df.copy()
# train_df_copy['Age'].fillna(train_df_copy['Age'].median(), inplace=True)
# eval_df_copy['Age'].fillna(eval_df_copy['Age'].median(), inplace=True)
# train_df_copy['Embarked'].fillna(train_df_copy['Embarked'].mode()[0], inplace=True)
# eval_df_copy['Embarked'].fillna(eval_df_copy['Embarked'].mode()[0], inplace=True)
# train_df_copy.drop(columns=['Cabin'], inplace=True)
# eval_df_copy.drop(columns=['Cabin'], inplace=True)
# label_encoder = LabelEncode(features=['Sex', 'Embarked'])
# train_df_copy = label_encoder.fit_transform(train_df_copy)
# eval_df_copy = label_encoder.transform(eval_df_copy)
# numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
# scaler = StandardScale(features=numerical_features)
# train_df_copy = scaler.fit_transform(train_df_copy)
# eval_df_copy = scaler.transform(eval_df_copy)
# train_df_copy['Survived'] = train_df['Survived']
# print(train_df_copy.head())
# print(eval_df_copy.head())
#
# import pandas as pd
# import numpy as np
# from thinker_ai.agent.tools.libs.feature_engineering import TargetMeanEncoder, GroupStat, CatCount, CatCross
# from thinker_ai.agent.tools.libs.data_preprocess import StandardScale, LabelEncode
# train_data_path = '../data/titanic/train.csv'
# eval_data_path = '../data/titanic/test.csv'
# train_df = pd.read_csv(train_data_path)
# eval_df = pd.read_csv(eval_data_path)
# train_df_copy = train_df.copy()
# eval_df_copy = eval_df.copy()
# train_df_copy['Age'] = train_df_copy['Age'].fillna(train_df_copy['Age'].median())
# eval_df_copy['Age'] = eval_df_copy['Age'].fillna(eval_df_copy['Age'].median())
# train_df_copy['Embarked'] = train_df_copy['Embarked'].fillna(train_df_copy['Embarked'].mode()[0])
# eval_df_copy['Embarked'] = eval_df_copy['Embarked'].fillna(eval_df_copy['Embarked'].mode()[0])
# train_df_copy = train_df_copy.drop(columns=['Cabin'])
# eval_df_copy = eval_df_copy.drop(columns=['Cabin'])
# label_encoder = LabelEncode(features=['Sex', 'Embarked'])
# train_df_copy = label_encoder.fit_transform(train_df_copy)
# eval_df_copy = label_encoder.transform(eval_df_copy)
# numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
# scaler = StandardScale(features=numerical_features)
# train_df_copy = scaler.fit_transform(train_df_copy)
# eval_df_copy = scaler.transform(eval_df_copy)
# train_df_copy['Survived'] = train_df['Survived']
# target_mean_encoder = TargetMeanEncoder(col='Pclass', label='Survived')
# train_df_copy = target_mean_encoder.fit_transform(train_df_copy)
# eval_df_copy = target_mean_encoder.transform(eval_df_copy)
# group_stat = GroupStat(group_col='Pclass', agg_col='Fare', agg_funcs=['mean', 'std'])
# train_df_copy = group_stat.fit_transform(train_df_copy)
# eval_df_copy = group_stat.transform(eval_df_copy)
# cat_count = CatCount(col='Ticket')
# train_df_copy = cat_count.fit_transform(train_df_copy)
# eval_df_copy = cat_count.transform(eval_df_copy)
# cat_cross = CatCross(cols=['Sex', 'Embarked'])
# train_df_copy = cat_cross.fit_transform(train_df_copy)
# eval_df_copy = cat_cross.transform(eval_df_copy)
# train_df_copy = train_df_copy.drop(columns=['PassengerId', 'Name'])
# eval_df_copy = eval_df_copy.drop(columns=['PassengerId', 'Name'])
# print(train_df_copy.head())
# print(eval_df_copy.head())
#     ```
#
#     ### execution result
#     <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 891 entries, 0 to 890
# Data columns (total 12 columns):
#  #   Column       Non-Null Count  Dtype
# ---  ------       --------------  -----
#  0   PassengerId  891 non-null    int64
#  1   Survived     891 non-null    int64
#  2   Pclass       891 non-null    int64
#  3   Name         891 non-null    object
#  4   Sex          891 non-null    object
#  5   Age          714 non-null    float64
#  6   SibSp        891 non-null    int64
#  7   Parch        891 non-null    int64
#  8   Ticket       891 non-null    object
#  9   Fare         891 non-null    float64
#  10  Cabin        204 non-null    object
#  11  Embarked     889 non-null    object
# dtypes: float64(2), int64(5), object(5)
# memory usage: 83.7+ KB
# Numerical columns: ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
# Categorical columns: ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
#        PassengerId    Survived      Pclass         Age       SibSp  \
# count   891.000000  891.000000  891.000000  714.000000  891.000000
# mean    446.000000    0.383838    2.308642   29.699118    0.523008
# std     257.353842    0.486592    0.836071   14.526497    1.102743
# min       1.000000    0.000000    1.000000    0.420000    0.000000
# 25%     223.500000    0.000000    2.000000   20.125000    0.000000
# 50%     446.000000    0.000000    3.000000   28.000000    0.000000
# 75%     668.500000    1.000000    3.000000   38.000000    1.000000
# max     891.000000    1.000000    3.000000   80.000000    8.000000
#
#             Parch        Fare
# count  891.000000  891.000000
# mean     0.381594   32.204208
# std      0.806057   49.693429
# min      0.000000    0.000000
# 25%      0.000000    7.910400
# 50%      0.000000   14.454200
# 75%      0.000000   31.000000
# max      6.000000  512.329200
#                            Name   Sex  Ticket    Cabin Embarked
# count                       891   891     891      204      889
# unique                      891     2
#
#    PassengerId  Survived    Pclass  \
# 0            1         0  0.827377
# 1            2         1 -1.566107
# 2            3         1  0.827377
# 3            4         1 -1.566107
# 4            5         0  0.827377
#
#                                                 Name  Sex       Age     SibSp  \
# 0                            Braund, Mr. Owen Harris    1 -0.565736  0.432793
# 1  Cumings, Mrs. John Bradley (Florence Briggs Th...    0  0.663861  0.432793
# 2                             Heikkinen, Miss. Laina    0 -0.258337 -0.474545
# 3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    0  0.433312  0.432793
# 4                           Allen, Mr. William Henry    1  0.433312 -0.474545
#
#       Parch            Ticket      Fare  Embarked
# 0 -0.473674         A/5 21171 -0.502445         2
# 1 -0.473674          PC 17599  0.786845         0
# 2 -0.473674  STON/O2. 3101282 -0.488854         2
# 3 -0.473674            113803  0.420730         2
# 4 -0.473674            373450 -0.486337         2
#    PassengerId    Pclass                                          Name  Sex  \
# 0          892  0.827377                              Kelly, Mr. James    1
# 1          893  0.827377              Wilkes, Mrs. James (Ellen Needs)    0
# 2          894 -0.369365                     Myles, Mr. Thomas Francis    1
# 3          895  0.827377                              Wirz, Mr. Albert    1
# 4          896  0.827377  Hirvonen, Mrs. Alexander (Helga E Lindqvist)    0
#
#         Age     SibSp     Parch   Ticket      Fare  Embarked
# 0  0.394887 -0.474545 -0.473674   330911 -0.490783         1
# 1  1.355510  0.432793 -0.473674   363272 -0.507479         2
# 2  2.508257 -0.474545 -0.473674   240276 -0.453367         1
# 3 -0.181487 -0.474545 -0.473674   315154 -0.474005         2
# 4 -0.565736  0.432793  0.767630  3101298 -0.401017         2
# ,/var/folders/dc/8m6ss95s4ml32zmx3_dd3n1m0000gn/T/ipykernel_3330/1917668986.py:18: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
# The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
#
# For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
#
#
#   train_df_copy['Age'].fillna(train_df_copy['Age'].median(), inplace=True)
# /var/folders/dc/8m6ss95s4ml32zmx3_dd3n1m0000gn/T/ipykernel_3330/1917668986.py:19: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
# The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
#
# For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
#
#
#   eval_df_copy['Age'].fillna(eval_df_copy['Age'].median(), inplace=True)
# /var/folders/dc/8m6ss95s4ml32zmx3_dd3n1m0000gn/T/ipykernel_3330/1917668986.py:22: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
# The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
#
# For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
#
#
#   train_df_copy['Embarked'].fillna(train_df_copy['Embarked'].mode()[0]
#
#    Survived    Pclass  Sex       Age     SibSp     Parch            Ticket  \
# 0         0  0.827377    1 -0.565736  0.432793 -0.473674         A/5 21171
# 1         1 -1.566107    0  0.663861  0.432793 -0.473674          PC 17599
# 2         1  0.827377    0 -0.258337 -0.474545 -0.473674  STON/O2. 3101282
# 3         1 -1.566107    0  0.433312  0.432793 -0.473674            113803
# 4         0  0.827377    1  0.433312 -0.474545 -0.473674            373450
#
#        Fare  Embarked  Pclass_target_mean  Fare_mean_by_Pclass  \
# 0 -0.502445         2            0.242363            -0.373069
# 1  0.786845         0            0.629630             1.046007
# 2 -0.488854         2            0.242363            -0.373069
# 3  0.420730         2            0.629630             1.046007
# 4 -0.486337         2            0.242363            -0.373069
#
#    Fare_std_by_Pclass  Ticket_cnt  Sex_Embarked
# 0            0.237149           1             0
# 1            1.578164           1             4
# 2            0.237149           1             3
# 3            1.578164           2             3
# 4            0.237149           1             0
#      Pclass  Sex       Age     SibSp     Parch   Ticket      Fare  Embarked  \
# 0  0.827377    1  0.394887 -0.474545 -0.473674   330911 -0.490783         1
# 1  0.827377    0  1.355510  0.432793 -0.473674   363272 -0.507479         2
# 2 -0.369365    1  2.508257 -0.474545 -0.473674   240276 -0.453367         1
# 3  0.827377    1 -0.181487 -0.474545 -0.473674   315154 -0.474005         2
# 4  0.827377    0 -0.565736  0.432793  0.767630  3101298 -0.401017         2
#
#    Pclass_target_mean  Fare_mean_by_Pclass  Fare_std_by_Pclass  Ticket_cnt  \
# 0            0.242363            -0.373069            0.237149         NaN
# 1            0.242363            -0.373069            0.237149         NaN
# 2            0.472826            -0.232395            0.270155         NaN
# 3            0.242363            -0.373069     ,/Users/wangli/Library/Mobile Documents/com~apple~CloudDocs/历史项目/projects/thinker-ai/src/thinker_ai/agent/tools/libs/feature_engineering.py:213: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
# The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
#
# For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
#
#
#   new_df[new_col].fillna(max(_map.values()) + 1, inplace=True)
# /Users/wangli/Library/Mobile Documents/com~apple~CloudDocs/历史项目/projects/thinker-ai/src/thinker_ai/agent/tools/libs/feature_engineering.py:213: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
# The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.
#
# For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.
#
#
#   new_df[new_col].fillna(max(_map.values()) + 1, inplace=True)
#
#
#     ## Current Task
#     Train a model to predict passenger survival
#
#     ## Task Guidance
#     Write complete code for 'Current Task'. And avoid duplicating code from 'Finished Tasks', such as repeated import of packages, reading data, etc.
#     Specifically,
# The current task is about training a model, please ensure high performance:
# - Keep in mind that your user prioritizes results and is highly focused on model performance. So, when needed, feel free to use models of any complexity to improve effectiveness, such as XGBoost, CatBoost, etc.
# - If non-numeric columns exist, perform label encode together with all steps.
# - Use the data from previous task result directly, do not mock or reload data yourself.
# - Set suitable hyperparameters for the model, make metrics as high as possible.
# """
#     tool_info = """
# ## Capabilities
# - You can utilize pre-defined tools in any code lines from 'Available Tools' in the form of Python class or function.
# - You can freely combine the use of any other public packages, like sklearn, numpy, pandas, etc..
#
# ## Available Tools:
# Each tool is described in JSON format. When you call a tool, import the tool from its path first.
# {'KFoldTargetMeanEncoder': {'type': 'class', 'description': 'Add a new feature to the DataFrame by k-fold mean encoding of a categorical column using the label column.', 'methods': {'__init__': {'type': 'function', 'description': 'Initialize self. ', 'signature': "(self, col: 'str', label: 'str', n_splits: 'int' = 5, random_state: 'int' = 2021)", 'parameters': 'Args: col (str): Column to be k-fold mean encoded. label (str): Predicted label column. n_splits (int, optional): Number of splits for K-fold. Defaults to 5. random_state (int, optional): Random seed. Defaults to 2021.'}, 'fit': {'type': 'function', 'description': 'Fit a model to be used in subsequent transform. ', 'signature': "(self, df: 'pd.DataFrame')", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame.'}, 'fit_transform': {'type': 'function', 'description': 'Fit and transform the input DataFrame. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}, 'transform': {'type': 'function', 'description': 'Transform the input DataFrame with the fitted model. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}}, 'tool_path': 'thinker_ai/agent/tools/libs/feature_engineering.py'}, 'LabelEncode': {'type': 'class', 'description': 'Apply label encoding to specified categorical columns in-place.', 'methods': {'__init__': {'type': 'function', 'description': 'Initialize self. ', 'signature': "(self, features: 'list')", 'parameters': 'Args: features (list): Categorical columns to be label encoded.'}, 'fit': {'type': 'function', 'description': 'Fit a model to be used in subsequent transform. ', 'signature': "(self, df: 'pd.DataFrame')", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame.'}, 'fit_transform': {'type': 'function', 'description': 'Fit and transform the input DataFrame. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}, 'transform': {'type': 'function', 'description': 'Transform the input DataFrame with the fitted model. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}}, 'tool_path': 'thinker_ai/agent/tools/libs/data_preprocess.py'}, 'OneHotEncode': {'type': 'class', 'description': 'Apply one-hot encoding to specified categorical columns, the original columns will be dropped.', 'methods': {'__init__': {'type': 'function', 'description': 'Initialize self. ', 'signature': "(self, features: 'list')", 'parameters': 'Args: features (list): Columns to be processed.'}, 'fit': {'type': 'function', 'description': 'Fit a model to be used in subsequent transform. ', 'signature': "(self, df: 'pd.DataFrame')", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame.'}, 'fit_transform': {'type': 'function', 'description': 'Fit and transform the input DataFrame. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}, 'transform': {'type': 'function', 'description': 'Transform the input DataFrame with the fitted model. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}}, 'tool_path': 'thinker_ai/agent/tools/libs/data_preprocess.py'}, 'StandardScale': {'type': 'class', 'description': 'Standardize features by removing the mean and scaling to unit variance.', 'methods': {'__init__': {'type': 'function', 'description': 'Initialize self. ', 'signature': "(self, features: 'list')", 'parameters': 'Args: features (list): Columns to be processed.'}, 'fit': {'type': 'function', 'description': 'Fit a model to be used in subsequent transform. ', 'signature': "(self, df: 'pd.DataFrame')", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame.'}, 'fit_transform': {'type': 'function', 'description': 'Fit and transform the input DataFrame. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}, 'transform': {'type': 'function', 'description': 'Transform the input DataFrame with the fitted model. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}}, 'tool_path': 'thinker_ai/agent/tools/libs/data_preprocess.py'}, 'VarianceBasedSelection': {'type': 'class', 'description': 'Select features based on variance and remove features with low variance.', 'methods': {'__init__': {'type': 'function', 'description': 'Initialize self. ', 'signature': "(self, label_col: 'str', threshold: 'float' = 0)", 'parameters': 'Args: label_col (str): Label column name. threshold (float, optional): Threshold for variance. Defaults to 0.'}, 'fit': {'type': 'function', 'description': 'Fit a model to be used in subsequent transform. ', 'signature': "(self, df: 'pd.DataFrame')", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame.'}, 'fit_transform': {'type': 'function', 'description': 'Fit and transform the input DataFrame. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}, 'transform': {'type': 'function', 'description': 'Transform the input DataFrame with the fitted model. ', 'signature': "(self, df: 'pd.DataFrame') -> 'pd.DataFrame'", 'parameters': 'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.'}}, 'tool_path': 'thinker_ai/agent/tools/libs/feature_engineering.py'}}
# """
#     context = [{'role': 'user',
#                 'content': '\n# User Requirement\nThis is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report accuracy on the eval data. Train data path: \'../data/titanic/train.csv\', eval data path: \'../data/titanic/test.csv\'.\n\n# Plan Status\n\n    ## Finished Tasks\n    ### code\n    ```python\n    import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\ntrain_data_path = \'../data/titanic/train.csv\'\ndf = pd.read_csv(train_data_path)\ndf.info()\ndf.head()\nnumerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()\ncategorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()\nprint(\'Numerical columns:\', numerical_cols)\nprint(\'Categorical columns:\', categorical_cols)\nprint(df[numerical_cols].describe())\nprint(df[categorical_cols].describe())\nplt.figure(figsize=(10, 6))\nsns.heatmap(df.isnull(), cbar=False, cmap=\'viridis\')\nplt.title(\'Missing Values Heatmap\')\nplt.show()\nplt.figure(figsize=(12, 8))\ncorr_matrix = df[numerical_cols].corr()\nsns.heatmap(corr_matrix, annot=True, cmap=\'coolwarm\', vmin=-1, vmax=1)\nplt.title(\'Correlation Matrix\')\nplt.show()\nfor col in numerical_cols:\n    plt.figure(figsize=(10, 6))\n    sns.histplot(df[col].dropna(), kde=True)\n    plt.title(f\'Distribution of {col}\')\n    plt.show()\nfor col in categorical_cols:\n    plt.figure(figsize=(10, 6))\n    sns.countplot(x=col, data=df)\n    plt.title(f\'Count Plot of {col}\')\n    plt.show()\n\nimport pandas as pd\nimport numpy as np\nfrom thinker_ai.agent.tools.libs.feature_engineering import TargetMeanEncoder, GeneralSelection\nfrom thinker_ai.agent.tools.libs.data_preprocess import StandardScale, LabelEncode\ntrain_data_path = \'../data/titanic/train.csv\'\neval_data_path = \'../data/titanic/test.csv\'\ntrain_df = pd.read_csv(train_data_path)\neval_df = pd.read_csv(eval_data_path)\ntrain_df_copy = train_df.copy()\neval_df_copy = eval_df.copy()\ntrain_df_copy[\'Age\'].fillna(train_df_copy[\'Age\'].median(), inplace=True)\neval_df_copy[\'Age\'].fillna(eval_df_copy[\'Age\'].median(), inplace=True)\ntrain_df_copy[\'Embarked\'].fillna(train_df_copy[\'Embarked\'].mode()[0], inplace=True)\neval_df_copy[\'Embarked\'].fillna(eval_df_copy[\'Embarked\'].mode()[0], inplace=True)\ntrain_df_copy.drop(columns=[\'Cabin\'], inplace=True)\neval_df_copy.drop(columns=[\'Cabin\'], inplace=True)\nlabel_encoder = LabelEncode(features=[\'Sex\', \'Embarked\'])\ntrain_df_copy = label_encoder.fit_transform(train_df_copy)\neval_df_copy = label_encoder.transform(eval_df_copy)\nnumerical_features = [\'Age\', \'Fare\', \'Pclass\', \'SibSp\', \'Parch\']\nscaler = StandardScale(features=numerical_features)\ntrain_df_copy = scaler.fit_transform(train_df_copy)\neval_df_copy = scaler.transform(eval_df_copy)\ntrain_df_copy[\'Survived\'] = train_df[\'Survived\']\nprint(train_df_copy.head())\nprint(eval_df_copy.head())\n\nimport pandas as pd\nimport numpy as np\nfrom thinker_ai.agent.tools.libs.feature_engineering import TargetMeanEncoder, GroupStat, CatCount, CatCross\nfrom thinker_ai.agent.tools.libs.data_preprocess import StandardScale, LabelEncode\ntrain_data_path = \'../data/titanic/train.csv\'\neval_data_path = \'../data/titanic/test.csv\'\ntrain_df = pd.read_csv(train_data_path)\neval_df = pd.read_csv(eval_data_path)\ntrain_df_copy = train_df.copy()\neval_df_copy = eval_df.copy()\ntrain_df_copy[\'Age\'] = train_df_copy[\'Age\'].fillna(train_df_copy[\'Age\'].median())\neval_df_copy[\'Age\'] = eval_df_copy[\'Age\'].fillna(eval_df_copy[\'Age\'].median())\ntrain_df_copy[\'Embarked\'] = train_df_copy[\'Embarked\'].fillna(train_df_copy[\'Embarked\'].mode()[0])\neval_df_copy[\'Embarked\'] = eval_df_copy[\'Embarked\'].fillna(eval_df_copy[\'Embarked\'].mode()[0])\ntrain_df_copy = train_df_copy.drop(columns=[\'Cabin\'])\neval_df_copy = eval_df_copy.drop(columns=[\'Cabin\'])\nlabel_encoder = LabelEncode(features=[\'Sex\', \'Embarked\'])\ntrain_df_copy = label_encoder.fit_transform(train_df_copy)\neval_df_copy = label_encoder.transform(eval_df_copy)\nnumerical_features = [\'Age\', \'Fare\', \'Pclass\', \'SibSp\', \'Parch\']\nscaler = StandardScale(features=numerical_features)\ntrain_df_copy = scaler.fit_transform(train_df_copy)\neval_df_copy = scaler.transform(eval_df_copy)\ntrain_df_copy[\'Survived\'] = train_df[\'Survived\']\ntarget_mean_encoder = TargetMeanEncoder(col=\'Pclass\', label=\'Survived\')\ntrain_df_copy = target_mean_encoder.fit_transform(train_df_copy)\neval_df_copy = target_mean_encoder.transform(eval_df_copy)\ngroup_stat = GroupStat(group_col=\'Pclass\', agg_col=\'Fare\', agg_funcs=[\'mean\', \'std\'])\ntrain_df_copy = group_stat.fit_transform(train_df_copy)\neval_df_copy = group_stat.transform(eval_df_copy)\ncat_count = CatCount(col=\'Ticket\')\ntrain_df_copy = cat_count.fit_transform(train_df_copy)\neval_df_copy = cat_count.transform(eval_df_copy)\ncat_cross = CatCross(cols=[\'Sex\', \'Embarked\'])\ntrain_df_copy = cat_cross.fit_transform(train_df_copy)\neval_df_copy = cat_cross.transform(eval_df_copy)\ntrain_df_copy = train_df_copy.drop(columns=[\'PassengerId\', \'Name\'])\neval_df_copy = eval_df_copy.drop(columns=[\'PassengerId\', \'Name\'])\nprint(train_df_copy.head())\nprint(eval_df_copy.head())\n    ```\n\n    ### execution result\n    <class \'pandas.core.frame.DataFrame\'>\nRangeIndex: 891 entries, 0 to 890\nData columns (total 12 columns):\n #   Column       Non-Null Count  Dtype  \n---  ------       --------------  -----  \n 0   PassengerId  891 non-null    int64  \n 1   Survived     891 non-null    int64  \n 2   Pclass       891 non-null    int64  \n 3   Name         891 non-null    object \n 4   Sex          891 non-null    object \n 5   Age          714 non-null    float64\n 6   SibSp        891 non-null    int64  \n 7   Parch        891 non-null    int64  \n 8   Ticket       891 non-null    object \n 9   Fare         891 non-null    float64\n 10  Cabin        204 non-null    object \n 11  Embarked     889 non-null    object \ndtypes: float64(2), int64(5), object(5)\nmemory usage: 83.7+ KB\nNumerical columns: [\'PassengerId\', \'Survived\', \'Pclass\', \'Age\', \'SibSp\', \'Parch\', \'Fare\']\nCategorical columns: [\'Name\', \'Sex\', \'Ticket\', \'Cabin\', \'Embarked\']\n       PassengerId    Survived      Pclass         Age       SibSp  \\\ncount   891.000000  891.000000  891.000000  714.000000  891.000000   \nmean    446.000000    0.383838    2.308642   29.699118    0.523008   \nstd     257.353842    0.486592    0.836071   14.526497    1.102743   \nmin       1.000000    0.000000    1.000000    0.420000    0.000000   \n25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n75%     668.500000    1.000000    3.000000   38.000000    1.000000   \nmax     891.000000    1.000000    3.000000   80.000000    8.000000   \n\n            Parch        Fare  \ncount  891.000000  891.000000  \nmean     0.381594   32.204208  \nstd      0.806057   49.693429  \nmin      0.000000    0.000000  \n25%      0.000000    7.910400  \n50%      0.000000   14.454200  \n75%      0.000000   31.000000  \nmax      6.000000  512.329200  \n                           Name   Sex  Ticket    Cabin Embarked\ncount                       891   891     891      204      889\nunique                      891     2  \n\n   PassengerId  Survived    Pclass  \\\n0            1         0  0.827377   \n1            2         1 -1.566107   \n2            3         1  0.827377   \n3            4         1 -1.566107   \n4            5         0  0.827377   \n\n                                                Name  Sex       Age     SibSp  \\\n0                            Braund, Mr. Owen Harris    1 -0.565736  0.432793   \n1  Cumings, Mrs. John Bradley (Florence Briggs Th...    0  0.663861  0.432793   \n2                             Heikkinen, Miss. Laina    0 -0.258337 -0.474545   \n3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    0  0.433312  0.432793   \n4                           Allen, Mr. William Henry    1  0.433312 -0.474545   \n\n      Parch            Ticket      Fare  Embarked  \n0 -0.473674         A/5 21171 -0.502445         2  \n1 -0.473674          PC 17599  0.786845         0  \n2 -0.473674  STON/O2. 3101282 -0.488854         2  \n3 -0.473674            113803  0.420730         2  \n4 -0.473674            373450 -0.486337         2  \n   PassengerId    Pclass                                          Name  Sex  \\\n0          892  0.827377                              Kelly, Mr. James    1   \n1          893  0.827377              Wilkes, Mrs. James (Ellen Needs)    0   \n2          894 -0.369365                     Myles, Mr. Thomas Francis    1   \n3          895  0.827377                              Wirz, Mr. Albert    1   \n4          896  0.827377  Hirvonen, Mrs. Alexander (Helga E Lindqvist)    0   \n\n        Age     SibSp     Parch   Ticket      Fare  Embarked  \n0  0.394887 -0.474545 -0.473674   330911 -0.490783         1  \n1  1.355510  0.432793 -0.473674   363272 -0.507479         2  \n2  2.508257 -0.474545 -0.473674   240276 -0.453367         1  \n3 -0.181487 -0.474545 -0.473674   315154 -0.474005         2  \n4 -0.565736  0.432793  0.767630  3101298 -0.401017         2  \n,/var/folders/dc/8m6ss95s4ml32zmx3_dd3n1m0000gn/T/ipykernel_3330/1917668986.py:18: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\nThe behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n\nFor example, when doing \'df[col].method(value, inplace=True)\', try using \'df.method({col: value}, inplace=True)\' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n\n\n  train_df_copy[\'Age\'].fillna(train_df_copy[\'Age\'].median(), inplace=True)\n/var/folders/dc/8m6ss95s4ml32zmx3_dd3n1m0000gn/T/ipykernel_3330/1917668986.py:19: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\nThe behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n\nFor example, when doing \'df[col].method(value, inplace=True)\', try using \'df.method({col: value}, inplace=True)\' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n\n\n  eval_df_copy[\'Age\'].fillna(eval_df_copy[\'Age\'].median(), inplace=True)\n/var/folders/dc/8m6ss95s4ml32zmx3_dd3n1m0000gn/T/ipykernel_3330/1917668986.py:22: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\nThe behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n\nFor example, when doing \'df[col].method(value, inplace=True)\', try using \'df.method({col: value}, inplace=True)\' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n\n\n  train_df_copy[\'Embarked\'].fillna(train_df_copy[\'Embarked\'].mode()[0]\n\n   Survived    Pclass  Sex       Age     SibSp     Parch            Ticket  \\\n0         0  0.827377    1 -0.565736  0.432793 -0.473674         A/5 21171   \n1         1 -1.566107    0  0.663861  0.432793 -0.473674          PC 17599   \n2         1  0.827377    0 -0.258337 -0.474545 -0.473674  STON/O2. 3101282   \n3         1 -1.566107    0  0.433312  0.432793 -0.473674            113803   \n4         0  0.827377    1  0.433312 -0.474545 -0.473674            373450   \n\n       Fare  Embarked  Pclass_target_mean  Fare_mean_by_Pclass  \\\n0 -0.502445         2            0.242363            -0.373069   \n1  0.786845         0            0.629630             1.046007   \n2 -0.488854         2            0.242363            -0.373069   \n3  0.420730         2            0.629630             1.046007   \n4 -0.486337         2            0.242363            -0.373069   \n\n   Fare_std_by_Pclass  Ticket_cnt  Sex_Embarked  \n0            0.237149           1             0  \n1            1.578164           1             4  \n2            0.237149           1             3  \n3            1.578164           2             3  \n4            0.237149           1             0  \n     Pclass  Sex       Age     SibSp     Parch   Ticket      Fare  Embarked  \\\n0  0.827377    1  0.394887 -0.474545 -0.473674   330911 -0.490783         1   \n1  0.827377    0  1.355510  0.432793 -0.473674   363272 -0.507479         2   \n2 -0.369365    1  2.508257 -0.474545 -0.473674   240276 -0.453367         1   \n3  0.827377    1 -0.181487 -0.474545 -0.473674   315154 -0.474005         2   \n4  0.827377    0 -0.565736  0.432793  0.767630  3101298 -0.401017         2   \n\n   Pclass_target_mean  Fare_mean_by_Pclass  Fare_std_by_Pclass  Ticket_cnt  \\\n0            0.242363            -0.373069            0.237149         NaN   \n1            0.242363            -0.373069            0.237149         NaN   \n2            0.472826            -0.232395            0.270155         NaN   \n3            0.242363            -0.373069     ,/Users/wangli/Library/Mobile Documents/com~apple~CloudDocs/历史项目/projects/thinker-ai/src/thinker_ai/agent/tools/libs/feature_engineering.py:213: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\nThe behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n\nFor example, when doing \'df[col].method(value, inplace=True)\', try using \'df.method({col: value}, inplace=True)\' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n\n\n  new_df[new_col].fillna(max(_map.values()) + 1, inplace=True)\n/Users/wangli/Library/Mobile Documents/com~apple~CloudDocs/历史项目/projects/thinker-ai/src/thinker_ai/agent/tools/libs/feature_engineering.py:213: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\nThe behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n\nFor example, when doing \'df[col].method(value, inplace=True)\', try using \'df.method({col: value}, inplace=True)\' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n\n\n  new_df[new_col].fillna(max(_map.values()) + 1, inplace=True)\n\n\n    ## Current Task\n    Train a model to predict passenger survival\n\n    ## Task Guidance\n    Write complete code for \'Current Task\'. And avoid duplicating code from \'Finished Tasks\', such as repeated import of packages, reading data, etc.\n    Specifically, \nThe current task is about training a model, please ensure high performance:\n- Keep in mind that your user prioritizes results and is highly focused on model performance. So, when needed, feel free to use models of any complexity to improve effectiveness, such as XGBoost, CatBoost, etc.\n- If non-numeric columns exist, perform label encode together with all steps.\n- Use the data from previous task result directly, do not mock or reload data yourself.\n- Set suitable hyperparameters for the model, make metrics as high as possible.\n\n    \n\n# Tool Info\n\n## Capabilities\n- You can utilize pre-defined tools in any code lines from \'Available Tools\' in the form of Python class or function.\n- You can freely combine the use of any other public packages, like sklearn, numpy, pandas, etc..\n\n## Available Tools:\nEach tool is described in JSON format. When you call a tool, import the tool from its path first.\n{\'KFoldTargetMeanEncoder\': {\'type\': \'class\', \'description\': \'Add a new feature to the DataFrame by k-fold mean encoding of a categorical column using the label column.\', \'methods\': {\'__init__\': {\'type\': \'function\', \'description\': \'Initialize self. \', \'signature\': "(self, col: \'str\', label: \'str\', n_splits: \'int\' = 5, random_state: \'int\' = 2021)", \'parameters\': \'Args: col (str): Column to be k-fold mean encoded. label (str): Predicted label column. n_splits (int, optional): Number of splits for K-fold. Defaults to 5. random_state (int, optional): Random seed. Defaults to 2021.\'}, \'fit\': {\'type\': \'function\', \'description\': \'Fit a model to be used in subsequent transform. \', \'signature\': "(self, df: \'pd.DataFrame\')", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame.\'}, \'fit_transform\': {\'type\': \'function\', \'description\': \'Fit and transform the input DataFrame. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}, \'transform\': {\'type\': \'function\', \'description\': \'Transform the input DataFrame with the fitted model. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}}, \'tool_path\': \'thinker_ai/agent/tools/libs/feature_engineering.py\'}, \'LabelEncode\': {\'type\': \'class\', \'description\': \'Apply label encoding to specified categorical columns in-place.\', \'methods\': {\'__init__\': {\'type\': \'function\', \'description\': \'Initialize self. \', \'signature\': "(self, features: \'list\')", \'parameters\': \'Args: features (list): Categorical columns to be label encoded.\'}, \'fit\': {\'type\': \'function\', \'description\': \'Fit a model to be used in subsequent transform. \', \'signature\': "(self, df: \'pd.DataFrame\')", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame.\'}, \'fit_transform\': {\'type\': \'function\', \'description\': \'Fit and transform the input DataFrame. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}, \'transform\': {\'type\': \'function\', \'description\': \'Transform the input DataFrame with the fitted model. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}}, \'tool_path\': \'thinker_ai/agent/tools/libs/data_preprocess.py\'}, \'OneHotEncode\': {\'type\': \'class\', \'description\': \'Apply one-hot encoding to specified categorical columns, the original columns will be dropped.\', \'methods\': {\'__init__\': {\'type\': \'function\', \'description\': \'Initialize self. \', \'signature\': "(self, features: \'list\')", \'parameters\': \'Args: features (list): Columns to be processed.\'}, \'fit\': {\'type\': \'function\', \'description\': \'Fit a model to be used in subsequent transform. \', \'signature\': "(self, df: \'pd.DataFrame\')", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame.\'}, \'fit_transform\': {\'type\': \'function\', \'description\': \'Fit and transform the input DataFrame. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}, \'transform\': {\'type\': \'function\', \'description\': \'Transform the input DataFrame with the fitted model. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}}, \'tool_path\': \'thinker_ai/agent/tools/libs/data_preprocess.py\'}, \'StandardScale\': {\'type\': \'class\', \'description\': \'Standardize features by removing the mean and scaling to unit variance.\', \'methods\': {\'__init__\': {\'type\': \'function\', \'description\': \'Initialize self. \', \'signature\': "(self, features: \'list\')", \'parameters\': \'Args: features (list): Columns to be processed.\'}, \'fit\': {\'type\': \'function\', \'description\': \'Fit a model to be used in subsequent transform. \', \'signature\': "(self, df: \'pd.DataFrame\')", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame.\'}, \'fit_transform\': {\'type\': \'function\', \'description\': \'Fit and transform the input DataFrame. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}, \'transform\': {\'type\': \'function\', \'description\': \'Transform the input DataFrame with the fitted model. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}}, \'tool_path\': \'thinker_ai/agent/tools/libs/data_preprocess.py\'}, \'VarianceBasedSelection\': {\'type\': \'class\', \'description\': \'Select features based on variance and remove features with low variance.\', \'methods\': {\'__init__\': {\'type\': \'function\', \'description\': \'Initialize self. \', \'signature\': "(self, label_col: \'str\', threshold: \'float\' = 0)", \'parameters\': \'Args: label_col (str): Label column name. threshold (float, optional): Threshold for variance. Defaults to 0.\'}, \'fit\': {\'type\': \'function\', \'description\': \'Fit a model to be used in subsequent transform. \', \'signature\': "(self, df: \'pd.DataFrame\')", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame.\'}, \'fit_transform\': {\'type\': \'function\', \'description\': \'Fit and transform the input DataFrame. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}, \'transform\': {\'type\': \'function\', \'description\': \'Transform the input DataFrame with the fitted model. \', \'signature\': "(self, df: \'pd.DataFrame\') -> \'pd.DataFrame\'", \'parameters\': \'Args: df (pd.DataFrame): The input DataFrame. Returns: pd.DataFrame: The transformed DataFrame.\'}}, \'tool_path\': \'thinker_ai/agent/tools/libs/feature_engineering.py\'}}\n\n\n# Constraints\n- Take on Current Task if it is in Plan Status, otherwise, tackle User Requirement directly.\n- Ensure the output new code is executable in the same Jupyter notebook as the previous executed code.\n- Always prioritize using pre-defined tools for the same functionality.\n\n# Output\nWhile some concise thoughts are helpful, code is absolutely required. Always output one and only one code block in your response. Output code in the following format:\n```python\nyour code\n```\n'},
#                {'role': 'user',
#                 'content': "\n    # Latest Data Info\n    Latest data info after previous tasks:\n    column_info\n{'Category': ['Ticket'], 'Numeric': ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass_target_mean', 'Fare_mean_by_Pclass', 'Fare_std_by_Pclass', 'Ticket_cnt', 'Sex_Embarked'], 'Datetime': [], 'Others': []}\n\n    "},
#                {'role': 'user', 'content': 'code fail:Unterminated string starting at: line 3 column 22 (char 578)'},
#                {'role': 'user', 'content': 'code fail:Unterminated string starting at: line 3 column 22 (char 454)'},
#                {'role': 'user', 'content': 'code fail:Unterminated string starting at: line 3 column 22 (char 500)'}]
#
#     user_requirement = """This is a titanic passenger survival dataset, your goal is to predict passenger survival outcome. The target column is Survived. Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target. Report accuracy on the eval data. Train data path: '../data/titanic/train.csv', eval data path: '../data/titanic/test.csv'."""
#
#     working_memory_dict = [{"role": "user",
#                             "content": "\n    # Latest Data Info\n    Latest data info after previous tasks:\n    column_info\n{'Category': ['Ticket'], 'Numeric': ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass_target_mean', 'Fare_mean_by_Pclass', 'Fare_std_by_Pclass', 'Ticket_cnt', 'Sex_Embarked'], 'Datetime': [], 'Others': []}\n\n    "},
#                            {"role": "user",
#                             "content": "code fail:Unterminated string starting at: line 3 column 22 (char 578)"},
#                            {"role": "user",
#                             "content": "code fail:Unterminated string starting at: line 3 column 22 (char 454)"},
#                            {"role": "user",
#                             "content": "code fail:Unterminated string starting at: line 3 column 22 (char 500)"}]
#
#     working_memory: list[Message] = [Message(content=item['content'], role=item['role']) for item in
#                                      working_memory_dict]
#
#     new_code = await WriteAnalysisCode().run(
#         user_requirement=user_requirement,
#         plan_status=plan_status,
#         tool_info=tool_info,
#         working_memory=working_memory,
#         use_reflection=True,
#     )
