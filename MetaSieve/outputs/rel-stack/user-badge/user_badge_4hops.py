from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('badges', 'f2p_UserId', 'users'): [64, 0, 0, 0],
    ('comments', 'f2p_UserId', 'users'): [0, 0, 64, 0],
    ('postHistory', 'f2p_PostId', 'posts'): [0, 64, 0, 0],
    ('postHistory', 'f2p_UserId', 'users'): [64, 0, 64, 64],
    ('postLinks', 'f2p_RelatedPostId', 'posts'): [0, 0, 0, 64],
    ('posts', 'f2p_AcceptedAnswerId', 'posts'): [0, 0, 0, 64],
    ('posts', 'f2p_OwnerUserId', 'users'): [64, 0, 64, 0],
    ('posts', 'f2p_ParentId', 'posts'): [0, 64, 0, 64],
    ('posts', 'rev_f2p_AcceptedAnswerId', 'posts'): [0, 0, 0, 64],
    ('posts', 'rev_f2p_ParentId', 'posts'): [0, 0, 64, 0],
    ('posts', 'rev_f2p_PostId', 'postHistory'): [0, 64, 0, 0],
    ('users', 'rev_f2p_OwnerUserId', 'posts'): [0, 64, 64, 0],
    ('users', 'rev_f2p_UserId', 'badges'): [0, 64, 0, 0],
    ('users', 'rev_f2p_UserId', 'postHistory'): [0, 64, 64, 64],
    ('votes', 'f2p_PostId', 'posts'): [0, 0, 64, 64],
    ('votes', 'f2p_UserId', 'users'): [0, 0, 64, 0],
}, default=[0] * 4)
