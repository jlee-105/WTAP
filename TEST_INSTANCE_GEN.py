
import pandas as pd
import random
import numpy as np

# Create instance_list
instance_list = []
for test_index in range(100):
    n_weapon = 50
    n_target = 50
    t_value = [random.randint(1, 10) for i in range(n_target)]
    #p = [[round(random.random(), 4) for j in range(tem_target)] for i in range(tem_weapon)]
    P = np.random.uniform(low=0.2, high=0.7, size=(n_weapon, n_target))
    #P_string = np.array2string(P, separator=', ')
    #tem_tw = [random.randint(0, 2) for i in range(tem_target)]
    TIME_START = [random.randint(0, 5) for _ in range(n_target)]
    time_window = list()
    for i in range(n_target):
        time_window.append((TIME_START[i], 10))

    instance_tuple = [n_weapon, n_target, t_value, P, time_window]
    instance_list.append(instance_tuple)

# Convert list to DataFrame without flattening
df = pd.DataFrame(instance_list, columns=['M', 'N', 'V', 'P', 'TW'])
import json
# Optionally, convert columns with nested lists to string representation
df['V'] = df['V'].apply(lambda x: str(x))
df['TW'] = df['TW'].apply(lambda x: str(x))
df['P'] = df['P'].apply(lambda x: json.dumps(x.tolist()))

df.to_excel("./TEST_INSTANCE/50M_50N_10T.xlsx")
