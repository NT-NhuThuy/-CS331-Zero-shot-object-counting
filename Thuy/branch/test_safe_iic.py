#!/usr/bin/env python3
import unittest
from safe_iic import safe_cluster

def c(cid,view='v0',q=1):return {'candidate_id':cid,'query_id':q,'view_id':view,'semantic':'YES','whole_instance':'YES'}

class SafeIICTests(unittest.TestCase):
    def test_bridge_edge_does_not_transitively_merge(self):
        xs=[c('a','v0'),c('b','v1'),c('c','v2')]
        out=safe_cluster(xs,[{'a':'a','b':'b','decision':'YES'},{'a':'b','b':'c','decision':'YES'},{'a':'a','b':'c','decision':'UNKNOWN'}])
        self.assertEqual(sorted(len(x['members']) for x in out['components']),[1,2])
    def test_clique_merges(self):
        xs=[c('a','v0'),c('b','v1'),c('c','v2')]
        e=[{'a':a,'b':b,'decision':'YES'} for a,b in (('a','b'),('a','c'),('b','c'))]
        out=safe_cluster(xs,e);self.assertEqual([x['members'] for x in out['components']],[['a','b','c']])
    def test_same_view_never_merges(self):
        xs=[c('a','v0'),c('b','v0')]
        out=safe_cluster(xs,[{'a':'a','b':'b','decision':'YES'}]);self.assertEqual(sorted(len(x['members']) for x in out['components']),[1,1])
    def test_overlapping_view_provenance_never_merges(self):
        xs=[{**c('a','v0'),'view_provenance':['v0','v1']},{**c('b','v2'),'view_provenance':['v1','v2']}]
        out=safe_cluster(xs,[{'a':'a','b':'b','decision':'YES'}]);self.assertEqual(sorted(len(x['members']) for x in out['components']),[1,1])
    def test_unknown_edge_is_not_merge(self):
        xs=[c('a','v0'),c('b','v1')]
        out=safe_cluster(xs,[{'a':'a','b':'b','decision':'UNKNOWN'}]);self.assertEqual(out['unresolved'][0]['status'],'UNKNOWN')
    def test_negative_edge_blocks_merge(self):
        xs=[c('a','v0'),c('b','v1')]
        out=safe_cluster(xs,[{'a':'a','b':'b','decision':'NO'}]);self.assertEqual(sorted(len(x['members']) for x in out['components']),[1,1])
    def test_large_component_is_diagnosed(self):
        xs=[c('a','v0'),c('b','v1')]
        out=safe_cluster(xs,[{'a':'a','b':'b','decision':'YES'}],large_component_threshold=2)
        self.assertEqual(out['large_components'][0]['size'],2)

if __name__=='__main__':unittest.main()
