import os
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    pp = False
else:
    pp = True
cut_query = 2
rotp = False
det_size = True
c_head = False
if not c_head:det_size = False
fp32 = True
use_aug = True
use_cb = False
use_ext = False
get_same_value_count = False
