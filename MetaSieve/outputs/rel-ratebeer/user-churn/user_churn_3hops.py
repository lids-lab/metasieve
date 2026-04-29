from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('availability', 'f2p_user_id', 'users'): [0, 0, 64],
    ('beer_ratings', 'f2p_user_id', 'users'): [64, 0, 64],
    ('beer_styles', 'rev_f2p_style_id', 'beers'): [0, 0, 64],
    ('beer_upcs', 'f2p_beer_id', 'beers'): [0, 0, 64],
    ('beers', 'rev_f2p_beer_id', 'beer_ratings'): [0, 64, 0],
    ('brewers', 'rev_f2p_brewer_id', 'beers'): [0, 0, 64],
    ('countries', 'rev_f2p_country_id', 'places'): [0, 0, 64],
    ('favorites', 'f2p_user_id', 'users'): [64, 0, 64],
    ('place_ratings', 'f2p_place_id', 'places'): [0, 0, 64],
    ('place_ratings', 'f2p_user_id', 'users'): [64, 0, 64],
    ('places', 'rev_f2p_place_id', 'place_ratings'): [0, 64, 0],
    ('users', 'rev_f2p_user_id', 'beer_ratings'): [0, 64, 0],
    ('users', 'rev_f2p_user_id', 'favorites'): [0, 64, 0],
    ('users', 'rev_f2p_user_id', 'place_ratings'): [0, 64, 0],
}, default=[0]*3)
