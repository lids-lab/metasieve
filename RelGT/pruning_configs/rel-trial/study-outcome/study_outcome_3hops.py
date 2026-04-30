from torch_geometric.sampler.base import NumNeighbors

num_neighbors = NumNeighbors({
    ('conditions_studies', 'f2p_nct_id', 'studies'): [0, 0, 64],
    ('designs', 'f2p_nct_id', 'studies'): [64, 0, 64],
    ('eligibilities', 'f2p_nct_id', 'studies'): [0, 0, 64],
    ('facilities', 'rev_f2p_facility_id', 'facilities_studies'): [0, 64, 0],
    ('facilities_studies', 'f2p_facility_id', 'facilities'): [0, 0, 64],
    ('facilities_studies', 'f2p_nct_id', 'studies'): [64, 0, 0],
    ('sponsors', 'rev_f2p_sponsor_id', 'sponsors_studies'): [0, 64, 0],
    ('sponsors_studies', 'f2p_nct_id', 'studies'): [64, 0, 64],
    ('sponsors_studies', 'f2p_sponsor_id', 'sponsors'): [0, 0, 64],
    ('studies', 'rev_f2p_nct_id', 'designs'): [0, 64, 0],
    ('studies', 'rev_f2p_nct_id', 'facilities_studies'): [0, 64, 0],
    ('studies', 'rev_f2p_nct_id', 'sponsors_studies'): [0, 64, 0],
}, default=[0]*3)
