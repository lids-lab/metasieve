from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('AdsInfo', 'f2p_LocationID', 'Location'): [0, 0, 64],
    ('AdsInfo', 'rev_f2p_AdID', 'SearchStream'): [0, 0, 64],
    ('AdsInfo', 'rev_f2p_AdID', 'VisitStream'): [0, 64, 0],
    ('Category', 'rev_f2p_CategoryID', 'AdsInfo'): [0, 0, 64],
    ('Location', 'rev_f2p_LocationID', 'AdsInfo'): [0, 0, 64],
    ('Location', 'rev_f2p_LocationID', 'SearchInfo'): [0, 64, 0],
    ('PhoneRequestsStream', 'f2p_UserID', 'UserInfo'): [64, 0, 64],
    ('SearchInfo', 'f2p_LocationID', 'Location'): [0, 0, 64],
    ('SearchInfo', 'f2p_UserID', 'UserInfo'): [64, 0, 64],
    ('SearchInfo', 'rev_f2p_SearchID', 'SearchStream'): [0, 0, 64],
    ('SearchStream', 'f2p_AdID', 'AdsInfo'): [0, 0, 64],
    ('SearchStream', 'f2p_SearchID', 'SearchInfo'): [0, 64, 0],
    ('UserInfo', 'rev_f2p_UserID', 'PhoneRequestsStream'): [0, 64, 0],
    ('UserInfo', 'rev_f2p_UserID', 'SearchInfo'): [0, 64, 0],
    ('UserInfo', 'rev_f2p_UserID', 'VisitStream'): [0, 64, 0],
    ('VisitStream', 'f2p_UserID', 'UserInfo'): [64, 0, 64],
}, default=[0]*3)
