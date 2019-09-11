import os, sys
import pandas as pd
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

from build_msg_net import build_msg_net
from msg_net_analysis import get_reach
from msg_net_analysis import reach_graph
from msg_net_analysis import influ_nodes
from msg_net_analysis import find_communities
from datetime import datetime

df = pd.read_csv('test_input/got_test_output_full_rev_v2.csv')
df_net = build_msg_net(df)
# results = get_reach(df_net, start_date = datetime(2019,3,1),
#     end_date = datetime(2019,4,1))
# result_df = reach_graph(df_net,start_date = datetime(2019,3,1),
#      end_date = datetime(2019,4,1))
# results = influ_nodes(df_net,df,start_date = datetime(2019,3,1),
#      end_date = datetime(2019,4,1))
results = influ_nodes(df_net,df)
for i,j in zip(results.keys(),results.values()):
    j.to_csv('test_output/'+ i + '.csv',header = True)

results2 = get_reach(df_net)

results3 = reach_graph(df_net, start_date = datetime(2019,1,1),
      end_date = datetime(2019,5,15))
results3.to_csv('test_output/reach_graph.csv')
# communities = find_communities(df_net,start_date = datetime(2019,3,1),
#       end_date = datetime(2019,4,1))
