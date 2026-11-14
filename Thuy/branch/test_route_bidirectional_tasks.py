import unittest
from route_bidirectional_tasks import route
class RoutingTests(unittest.TestCase):
 def test_rescue_always_and_dense_budget(self):
  xs=[{'task_id':f'b{i}','query_id':1,'role':'baseline','box_xyxy':[i*20,0,i*20+10,10],'proposal_score':i/10} for i in range(4)]+[{'task_id':'r','query_id':1,'role':'rescue','box_xyxy':[0,0,1,1]}]
  y=route(xs,2);self.assertEqual(sum(x['route_selected'] and x['role']=='baseline' for x in y),2);self.assertTrue(next(x for x in y if x['task_id']=='r')['route_selected'])
 def test_sparse_verifies_all(self):
  xs=[{'task_id':'b','query_id':1,'role':'baseline','box_xyxy':[0,0,1,1]}];self.assertTrue(route(xs,2)[0]['route_selected'])
