import pandas as pd

url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vS2YjVwU3IQAo2ITTvtN6rYqZjHiRYAiX0nH2wcRtnKzkbusE6OHWzkpyq8l6R8pybap_x4MhsJKuAK/pubhtml?gid=0&single=true'
# Link de la hoja https://docs.google.com/spreadsheets/d/1RrPyr4e3RGfm1XWAv-uoNHv63HBXgXwJanmetuLV1V0/edit#gid=0

df_list = pd.read_html(url)
df = pd.DataFrame(df_list[0])

df.columns = df.iloc[0]
df = df.iloc[1:]

print(df.head())
