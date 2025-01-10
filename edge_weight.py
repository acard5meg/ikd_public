import torch

def edge_weight_calculation(trav_map, elev_map, use_trav_map, normal_t, normal_e, normalize ):
    '''
    returns the edge weight when constructing the graph 

    trav_map : traversibility map from utils program
    elev_map : elevation map from utils program
    use_trav_map : boolean, which map to use 

    normal_t : denominator to normalize traversibility map edge weight
    normal_e : denominator to normalize elevation map edge weight
    normalize : boolean, whether to normalize edge weight with starting value

    returns: float
    '''
    edge_weight = 0

    if use_trav_map:
        edge_weight = torch.sum(trav_map).cuda().unsqueeze(0).item()

    else:
        back_l = torch.mean(elev_map[ : , : 4, : 4]).item()
        back_r = torch.mean(elev_map[ : , : 4, elev_map.shape[2]-4 : ]).item()
        front_l = torch.mean(elev_map[ : , elev_map.shape[2]-4 : , : 4]).item()
        front_r = torch.mean(elev_map[ : , elev_map.shape[2]-4 : , elev_map.shape[2] - 4 : ]).item()
        edge_weight = abs(back_l - back_r) + abs(front_l - front_r) + abs(front_l - back_l) + abs(front_r + back_r)

    if normalize and use_trav_map:
        edge_weight /= normal_t

    if normalize and not use_trav_map:
        edge_weight /= normal_e

    return edge_weight