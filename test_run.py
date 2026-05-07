import pandas as pd
from datascore import score

df = pd.read_csv(r"D:\Downloads\Projects\ml_prod_template\data\raw\telco_churn.csv")
report = score(df, target="Churn")
report.show()