import json

def prob_config_parser(fname):
    # * parse the problem configuration file
    """
    format: 
        {'robot': {'pose': pose, 'urdf': urdf},
         'table': {'pose': pose, 'urdf': urdf},
         'objects': [{'pose': pose, 'urdf': urdf}],
         'camera': {'pose': pose, 'urdf': urdf},
         'placement': [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]}
    """
    f = open(fname, 'r')
    data = json.load(f)
    return data
