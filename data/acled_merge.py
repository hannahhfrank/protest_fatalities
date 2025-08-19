import pandas as pd

# Merge data 
battles = pd.read_csv("1997-01-01-2023-12-31_battles.csv",low_memory=False)
force_protest = pd.read_csv("1997-01-01-2023-12-31_excessive_force_against_protesters.csv",low_memory=False)
df = pd.concat([battles, force_protest], ignore_index=True)
remote = pd.read_csv("1997-01-01-2023-12-31_explosives_remote_violence.csv",low_memory=False)
df = pd.concat([df, remote], ignore_index=True)
peace_protest = pd.read_csv("1997-01-01-2023-12-31_peaceful_protest.csv",low_memory=False)
df = pd.concat([df, peace_protest], ignore_index=True)
intervention_protest = pd.read_csv("1997-01-01-2023-12-31_protest_with_intervention.csv",low_memory=False)
df = pd.concat([df, intervention_protest], ignore_index=True)
riots = pd.read_csv("1997-01-01-2023-12-31_riots.csv",low_memory=False)
df = pd.concat([df, riots], ignore_index=True)
development = pd.read_csv("1997-01-01-2023-12-31_strategic_developments.csv",low_memory=False)
df = pd.concat([df, development], ignore_index=True)
civilian = pd.read_csv("1997-01-01-2023-12-31_violence_against_civilians.csv",low_memory=False)
df = pd.concat([df, civilian], ignore_index=True)


df.to_csv("acled_all_events.csv")  
print(df.duplicated().any())




