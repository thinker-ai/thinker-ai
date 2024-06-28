import pytest

from thinker_ai.agent.actions.di.execute_nb_code import ExecuteNbCode


@pytest.mark.asyncio
async def test_code_running():
    executor = ExecuteNbCode()
    try:
        output, is_success = await executor.run("print('hello world!')")
        assert "hello world!" in output
        assert is_success
    finally:
        await executor.reset()


@pytest.mark.asyncio
async def test_split_code_running():
    executor = ExecuteNbCode()
    try:
        _ = await executor.run("x=1\ny=2")
        _ = await executor.run("z=x+y")
        output, is_success = await executor.run("assert z==3")
        assert is_success
    finally:
        await executor.reset()


@pytest.mark.asyncio
async def test_execute_error():
    executor = ExecuteNbCode()
    try:
        code="""
        from thinker_ai.agent.tools.libs.feature_engineering import TargetMeanEncoder, OneHotEncode, CatCount, GroupStat, PolynomialExpansion

# Copy the preprocessed data
train_df_fe = train_df_copy.copy()
eval_df_fe = eval_df_copy.copy()

# Target Mean Encoding for 'Pclass'
target_mean_encoder = TargetMeanEncoder(col='Pclass', label='Survived')
train_df_fe = target_mean_encoder.fit_transform(train_df_fe)
eval_df_fe = target_mean_encoder.transform(eval_df_fe)

# One-Hot Encoding for 'Embarked'
one_hot_encoder = OneHotEncode(features=['Embarked'])
train_df_fe = one_hot_encoder.fit_transform(train_df_fe)
eval_df_fe = one_hot_encoder.transform(eval_df_fe)

# Adding value counts for 'Sex'
cat_count = CatCount(col='Sex')
train_df_fe = cat_count.fit_transform(train_df_fe)
eval_df_fe = cat_count.transform(eval_df_fe)

# Group statistics for 'Fare' by 'Pclass'
group_stat = GroupStat(group_col='Pclass', agg_col='Fare', agg_funcs=['mean', 'std'])
train_df_fe = group_stat.fit_transform(train_df_fe)
eval_df_fe = group_stat.transform(eval_df_fe)

# Polynomial Expansion for 'Age' and 'Fare'
polynomial_expansion = PolynomialExpansion(cols=['Age', 'Fare'], label_col='Survived', degree=2)
train_df_fe = polynomial_expansion.fit_transform(train_df_fe)
eval_df_fe = polynomial_expansion.transform(eval_df_fe)

print(train_df_fe.head())
print(eval_df_fe.head())
        """
        output, is_success = await executor.run(code)
        assert not is_success
    finally:
        await executor.reset()


PLOT_CODE = """
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
random_data = np.random.randn(1000)  # 生成1000个符合标准正态分布的随机数

# 绘制直方图
plt.hist(random_data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

# 添加标题和标签
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 显示图形
plt.show()
plt.close()
"""


@pytest.mark.asyncio
async def test_plotting_code():
    executor = ExecuteNbCode()
    try:
        output, is_success = await executor.run(PLOT_CODE)
        assert is_success
    finally:
        await executor.reset()


@pytest.mark.asyncio
async def test_run_with_timeout():
    executor = ExecuteNbCode(timeout=1)
    try:
        code = "import time; time.sleep(2)"
        message, success = await executor.run(code)
        assert not success
        assert message.startswith("Cell execution timed out")
    finally:
        await executor.reset()


@pytest.mark.asyncio
async def test_run_code_text():
    executor = ExecuteNbCode()
    try:
        message, success = await executor.run(code='print("This is a code!")', language="python")
        assert success
        assert "This is a code!" in message
        message, success = await executor.run(code="# This is a code!", language="markdown")
        assert success
        assert message == "# This is a code!"
        mix_text = "# Title!\n ```python\n print('This is a code!')```"
        message, success = await executor.run(code=mix_text, language="markdown")
        assert success
        assert message == mix_text
    finally:
        await executor.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "k", [(1), (5)]
)  # k=1 to test a single regular terminate, k>1 to test terminate under continuous run
async def test_terminate(k):
    for _ in range(k):
        executor = ExecuteNbCode()
        try:
            await executor.run(code='print("This is a code!")', language="python")
            assert executor.executor.kc is None
            assert executor.executor.km is None
        finally:
            await executor.reset()


@pytest.mark.asyncio
async def test_reset():
    executor = ExecuteNbCode()
    try:
        await executor.run(code='print("This is a code!")', language="python")
        await executor.reset()
        assert executor.executor.km is None
    finally:
        await executor.reset()


@pytest.mark.asyncio
async def test_parse_outputs():
    executor = ExecuteNbCode()
    try:
        code = """
        import pandas as pd
        df = pd.DataFrame({'ID': [1,2,3], 'NAME': ['a', 'b', 'c']})
        print(df.columns)
        print(f"columns num:{len(df.columns)}")
        print(df['DUMMPY_ID'])
        """
        output, is_success = await executor.run(code)
        assert not is_success
        assert "Index(['ID', 'NAME'], dtype='object')" in output
        assert "KeyError: 'DUMMPY_ID'" in output
        assert "columns num:2" in output
    finally:
        await executor.reset()
