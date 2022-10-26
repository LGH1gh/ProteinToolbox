import os
import sys
sys.path.append('../ProteinToolbox/')


from proteintoolbox.utils import fasta2dict

class TestPreprocess:
  def test_fasta2dict(self):
    name2sequence = fasta2dict('./test/data/8APK.fasta')
    assert name2sequence['8APK_1|Chains A[auth L], M[auth l]|subunit-e|Trypanosoma brucei brucei (5702)'] == 'MSAKAAPKTLHQVRNVAYFFAAWLGVQKGYIEKSANDRLWVEHQRKVRQQNVERQQALDSIKLMQQGVRATTPGQLEGVPAELQQLAEAFTK'
