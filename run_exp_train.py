from src.layers import *

pose = Pose(data_dir, device='cuda')

# ################## predictions on sepsis with high prob but low ppiu scores
# out_dir = '/global/project/hpcg1830/hao/PoSEPath/out/sepsis_lo/'


# query = pose.get_prediction_test()
# pred_df = query.get_pred_table()
# pred_df = pred_df.sort_values(by='prob', ascending=False)

# pred_df = pandas.read_csv('/global/home/hpc4590/share/PoSEPath/data/sepsis/sepsis_lo.csv')

# for i in range(pred_df.shape[0]):
#     df = pred_df.iloc[i, :].values
#     d1, d2, s = df[0], df[3], df[6]
#     q = pose.predict([d1], [d2], [s])

#     if_auto_tuning = True if q.ppiu_score[0] > 0.08 else False
#     q = pose.explain_query(q, if_auto_tuning=if_auto_tuning)
#     q.to_pickle(out_dir+f'{d1}-{d2}-{s}.pkl')


# for i in range(1, 21):
#     df = pred_df.iloc[-i, :].values
#     d1, d2, s = df[0], df[3], df[6]
#     q = pose.predict([d1], [d2], [s])

#     if_auto_tuning = True if q.ppiu_score[0] > 0.08 else False
#     q = pose.explain_query(q, if_auto_tuning=if_auto_tuning)
#     q.to_pickle(out_dir+f'{d1}-{d2}-{s}.pkl')
#     # print(i)

# ################################################################
# ################# explain for each side effect #################
# out_dir = '/global/project/hpcg1830/hao/PoSEPath/out/side_effect/'
# for i in range(604, 861):
#     a, b = gdata.train_range.tolist()[i]
#     d1, d2 = gdata.train_idx[:, a:b].tolist()
#     se = [i] * len(d1)
#     q = pose.explain_list(d1, d2, se)
    
#     q.to_pickle(out_dir+f'{se[0]}.pkl')
#     # print(i)

# ################################################################
# ################# explain for alzheimer #################
out_dir = '/global/project/hpcg1830/hao/PoSEPath/out/alzheimer/'
drugs = [82, 97, 102, 191]  # donepezil, memantine, citalopram, risperidone
for i in range(len(drugs)):
    mask = gdata.train_idx[0]==drugs[i]
    d1, d2 = gdata.train_idx[:, mask].tolist()
    se = gdata.train_et[mask].tolist()

    q = pose.explain_list(d1, d2, se)
    q.to_pickle(f'{out_dir}drug-{drugs[i]}.pkl')
    print(i)