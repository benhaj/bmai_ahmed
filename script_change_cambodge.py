#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read('/hdd/data/bmai_clean/full_cambodge_data.csv')
df.img = df.img.apply(lambda x: x.replace("\\","/"))
df.to_csv('/hdd/data/bmai_clean/full_cambodge_data.csv',index=False)

