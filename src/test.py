import numpy as np
import pandas as pd

# 使用pandas读取CSV，它会自动处理被引号包围的字段
df = pd.read_csv('/home/chenyingying/tmp/cav-mae-sync/src/data_info/vggsound.csv', 
                 names=['id', 'timestamp', 'label', 'split'],
                 header=None)

# check the item 100:
for i, row in df.iterrows():
    if i == 100:
        print(row)
        break