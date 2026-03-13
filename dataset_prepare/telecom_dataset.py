import pandas as pd
churn80 = pd.read_csv("../datasets/churn-bigml-80.csv")
churn20 = pd.read_csv("../datasets/churn-bigml-20.csv")
print(churn20)
print(churn80)

churn100=pd.concat([churn20, churn80])
churn100.to_csv('../datasets/telecom_churn100.csv', index=False)
