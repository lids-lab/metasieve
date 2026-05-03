from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('AdsInfo', 'rev_f2p_AdID', 'SearchStream'): [0, 0, 0, 64],
    ('AdsInfo', 'rev_f2p_AdID', 'VisitStream'): [0, 64, 0, 64],
    ('Category', 'rev_f2p_CategoryID', 'SearchInfo'): [0, 0, 0, 64],
    ('Location', 'rev_f2p_LocationID', 'SearchInfo'): [0, 0, 0, 64],
    ('PhoneRequestsStream', 'f2p_UserID', 'UserInfo'): [64, 0, 64, 0],
    ('SearchInfo', 'f2p_UserID', 'UserInfo'): [64, 0, 64, 0],
    ('SearchInfo', 'rev_f2p_SearchID', 'SearchStream'): [0, 0, 64, 64],
    ('SearchStream', 'f2p_AdID', 'AdsInfo'): [0, 0, 64, 0],
    ('SearchStream', 'f2p_SearchID', 'SearchInfo'): [0, 64, 0, 64],
    ('UserInfo', 'rev_f2p_UserID', 'PhoneRequestsStream'): [0, 64, 0, 64],
    ('UserInfo', 'rev_f2p_UserID', 'SearchInfo'): [0, 64, 0, 0],
    ('UserInfo', 'rev_f2p_UserID', 'VisitStream'): [0, 64, 0, 64],
    ('VisitStream', 'f2p_UserID', 'UserInfo'): [64, 0, 64, 0],
}, default=[0]*4)
