from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('availability', 'f2p_user_id', 'users'): [0, 0, 64, 0],
    ('beer_ratings', 'f2p_user_id', 'users'): [64, 0, 64, 0],
    ('beer_styles', 'rev_f2p_style_id', 'beers'): [0, 0, 64, 0],
    ('beer_upcs', 'f2p_beer_id', 'beers'): [0, 0, 64, 0],
    ('beers', 'f2p_brewer_id', 'brewers'): [0, 0, 0, 64],
    ('beers', 'f2p_style_id', 'beer_styles'): [0, 0, 0, 64],
    ('beers', 'rev_f2p_beer_id', 'availability'): [0, 0, 0, 64],
    ('beers', 'rev_f2p_beer_id', 'beer_ratings'): [0, 64, 0, 64],
    ('beers', 'rev_f2p_beer_id', 'beer_upcs'): [0, 0, 0, 64],
    ('brewers', 'rev_f2p_brewer_id', 'beers'): [0, 0, 64, 0],
    ('countries', 'rev_f2p_country_id', 'availability'): [0, 0, 0, 64],
    ('countries', 'rev_f2p_country_id', 'brewers'): [0, 0, 0, 64],
    ('favorites', 'f2p_user_id', 'users'): [0, 0, 64, 0],
    ('place_ratings', 'f2p_user_id', 'users'): [64, 0, 0, 0],
    ('places', 'rev_f2p_place_id', 'availability'): [0, 0, 0, 64],
    ('states', 'rev_f2p_state_id', 'brewers'): [0, 0, 0, 64],
    ('users', 'rev_f2p_user_id', 'beer_ratings'): [0, 64, 0, 0],
    ('users', 'rev_f2p_user_id', 'favorites'): [0, 0, 0, 64],
    ('users', 'rev_f2p_user_id', 'place_ratings'): [0, 64, 0, 0],
}, default=[0]*4)
