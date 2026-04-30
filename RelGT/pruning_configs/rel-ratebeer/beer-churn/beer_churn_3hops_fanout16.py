from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('availability', 'f2p_beer_id', 'beers'): [0, 0, 16],
    ('availability', 'f2p_user_id', 'users'): [0, 0, 16],
    ('availability', 'rev_f2p_availability_id', 'beer_ratings'): [0, 16, 0],
    ('beer_ratings', 'f2p_availability_id', 'availability'): [0, 0, 16],
    ('beer_ratings', 'f2p_beer_id', 'beers'): [16, 0, 16],
    ('beer_ratings', 'f2p_user_id', 'users'): [0, 0, 16],
    ('beer_upcs', 'f2p_beer_id', 'beers'): [16, 0, 16],
    ('beers', 'f2p_brewer_id', 'brewers'): [0, 16, 0],
    ('beers', 'rev_f2p_beer_id', 'beer_ratings'): [0, 16, 0],
    ('beers', 'rev_f2p_beer_id', 'beer_upcs'): [0, 16, 0],
    ('brewers', 'rev_f2p_brewer_id', 'beers'): [16, 0, 16],
    ('place_ratings', 'f2p_user_id', 'users'): [0, 0, 16],
    ('places', 'rev_f2p_place_id', 'availability'): [0, 0, 16],
    ('states', 'rev_f2p_state_id', 'brewers'): [0, 16, 0],
    ('users', 'rev_f2p_user_id', 'beer_ratings'): [0, 16, 0],
}, default=[0]*3)
