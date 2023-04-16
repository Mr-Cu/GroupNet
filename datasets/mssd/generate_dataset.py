import math
import pandas as pd
import numpy as np
import os

files = os.listdir("datasets")
index=0
for file in files:
    index+=1
    inner_path="datasets/"+file
    inner_file=os.listdir(inner_path)
    home_team=pd.read_csv(inner_path+"/"+inner_file[2])
    away_team=pd.read_csv(inner_path+"/"+inner_file[1])
    home_team.drop(home_team.index[:1],inplace=True)
    away_team.drop(away_team.index[:1],inplace=True)

#     tmp_team=home_team.iloc[:,1:3]
    tmp_ball=home_team.iloc[:,31:33]
    tmp_players_1=home_team.iloc[:,5:25]
    tmp_players_2=home_team.iloc[:,3:5]
    tmp_players_3=home_team.iloc[:,25:31]
    tmp_players_4=away_team.iloc[:,5:25]
    tmp_players_5=away_team.iloc[:,3:5]
    if(index==1):
        tmp_players_6=away_team.iloc[:,25:31]
    if(index==2):
        tmp_players_6=away_team.iloc[:,25:27]
    df = pd.concat([tmp_ball,tmp_players_1,tmp_players_2,tmp_players_3,tmp_players_4,tmp_players_5,tmp_players_6],axis=1)
    df.drop_duplicates()  #数据去重
    name='Sample_Game_'+str(index)+'_RawTrackingData.csv'
#     print(df)
    df.to_csv(name,index=False,header=False,encoding = 'utf-8')
    # print("success")

data1=[]
data2=[]
for i in range(1,3):
    name='Sample_Game_'+str(i)+'_RawTrackingData.csv'
    df=pd.read_csv(name,header=None)
    df.drop(df.index[0],inplace=True)
    moments_length=15
#     print(df)
    length=len(df)//moments_length
    for k in range(0,length-1):
        tmp_data=df[k*moments_length:k*moments_length+moments_length]
        flag=1
        for j in range(0,moments_length):
            if(tmp_data.iloc[j,0]!=tmp_data.iloc[j,0]):
                flag=0
        if(flag==1):
            if(i==1):
                data1.append(tmp_data)
            if(i==2):
                data2.append(tmp_data)
# print("success")

all_data=[]
# print(all_data)
for part in data1:
    part.dropna(axis=1, how='any',inplace=True)
    if(len(part.columns)==46):
        all_data.append(part)

for part in data2:
    part.dropna(axis=1, how='any',inplace=True)
    if(len(part.columns)==46):
        all_data.append(part)

# print(all_data)

all_clear_data=[]

for data in all_data:
    index=0
    for i in range(0,moments_length):
        for j in range(0,23):
            position_x=round(float(data.iloc[i,2*j]),5)
            position_y=round(float(data.iloc[i,2*j+1]),5)
            if(position_x>1 or position_x<0 or position_y<0 or position_y>1):
                index+=1
    if(index<4):
        all_clear_data.append(data)

# print("success")

# print(len(all_data))
all_events=[]
for data in all_clear_data:
    events=[]
    for i in range(0,moments_length):
        moments=[]
        for j in range(0,23):
#             print(data.iloc[i,j])
            position=[]
            position_x=round(float(data.iloc[i,2*j]),5)
            position_y=round(float(data.iloc[i,2*j+1]),5)
            if(position_x<0):
                position_x=0
            if(position_x>1):
                position_x=1
            if(position_y<0):
                position_y=0
            if(position_y>1):
                position_y=1
            position_x=position_x*105
            position_y=position_y*68
            position.append(position_x)
            position.append(position_y)
            moments.append(position)
#             print(position)
#         print(moments)
        events.append(moments)
#     print(events)
    all_events.append(events)
# print("success")
# print(all_events)

all_events = np.array(all_events,dtype=np.float32)
all_events = np.unique(all_events,axis=0)

# print(all_events)

index = list(range(len(all_events)))
from random import shuffle
shuffle(index)

train_set = all_events[index[:7500]]
test_set = all_events[index[7500:]]
print('train num:',train_set.shape[0])
print('test num:',test_set.shape[0])

np.save('train.npy',train_set)
np.save('test.npy',test_set)
