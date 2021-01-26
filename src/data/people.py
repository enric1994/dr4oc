import random
import math
from sklearn.datasets.samples_generator import make_blobs

from src.data.utils.places_utils import get_random_image
from src.data.utils.humans3d_utils import get_lowpoly_human

MIN_LIGHTS = 5
MAX_LIGHTS = 8
LIGHT_ENERGY_MIN = 0.4
LIGHT_ENERGY_MAX = 2
LIGHT_POSITION_VAR = 3
LIGHT_CAST_SHADOWS = True
LIGHT_TYPES = ['SUN']
LIGHT_COLOR_MIN = 1
LIGHT_NEGATIVE = False

OBJECT_SCALE_MIN = 0.1
OBJECT_SCALE_MAX = 0.1#0.2

OBJECT_TRANSFORMS = ['RANDOMIZE'] #['CAST', 'SOLIDIFY', 'SIMPLE_DEFORM', 'RANDOMIZE']
OBJECT_TRANSFORMS_MIN = 0
OBJECT_TRANSFORMS_MAX = 0

OBJECT_TRANSFORMS_CAST_FACTOR_MIN = -0.4
OBJECT_TRANSFORMS_CAST_FACTOR_MAX = 0.3
OBJECT_TRANSFORMS_CAST_TYPES = ['SPHERE', 'CUBOID']

OBJECT_TRANSFORMS_SOLIDIFY_THICKNESS_MIN = -0.18
OBJECT_TRANSFORMS_SOLIDIFY_THICKNESS_MAX = 0.4

OBJECT_TRANSFORMS_SIMPLE_DEFORM_ANGLE_MIN = -0.3
OBJECT_TRANSFORMS_SIMPLE_DEFORM_ANGLE_MAX = 0.3

OBJECT_TRANSFORMS_RANDOMIZE_FACTOR_MIN = -0.05
OBJECT_TRANSFORMS_RANDOMIZE_FACTOR_MAX = 0.05

BACKGROUND_CLASSES = ['cliff', 'hotel-outdoor', 'hangar-outdoor', 'bridge', 'moat-water', 'pond', 'hospital_room', 'mezzanine', 'ocean', 'apartment_building-outdoor', 'shoe_shop', 'bow_window-indoor', 'raceway', 'forest_path', 'ski_slope', 'building_facade', 'boathouse', 'hardware_store', 'barndoor', 'cafeteria', 'aqueduct', 'village', 'iceberg', 'lighthouse', 'discotheque', 'sky', 'alley', 'corral', 'tower', 'oast_house', 'amusement_park', 'balcony-exterior', 'slum', 'delicatessen', 'pasture', 'embassy', 'jacuzzi-indoor', 'balcony-interior', 'office_cubicles', 'pub-indoor', 'legislative_chamber', 'basketball_court-indoor', 'gymnasium-indoor', 'harbor', 'kitchen', 'chalet', 'watering_hole', 'shower', 'elevator_lobby', 'forest-broadleaf', 'beauty_salon', 'staircase', 'auditorium', 'train_interior', 'living_room', 'swimming_pool-outdoor', 'restaurant_kitchen', 'swamp', 'lecture_room', 'biology_laboratory', 'forest_road', 'heliport', 'gift_shop', 'bazaar-outdoor', 'ball_pit', 'hangar-indoor', 'canal-urban', 'bank_vault', 'hunting_lodge-outdoor', 'loading_dock', 'boxing_ring', 'entrance_hall', 'shed', 'mosque-outdoor', 'desert-vegetation', 'parking_lot', 'elevator_shaft', 'throne_room', 'kennel-outdoor', 'pantry', 'recreation_room', 'jail_cell', 'inn-outdoor', 'locker_room', 'lock_chamber', 'arcade', 'swimming_hole', 'cockpit', 'boat_deck', 'fire_station', 'mountain', 'chemistry_lab', 'rope_bridge', 'lagoon', 'airplane_cabin', 'downtown', 'tree_farm', 'islet', 'mountain_snowy', 'ruin', 'attic', 'ice_floe', 'car_interior', 'lake-natural', 'dorm_room', 'home_theater', 'greenhouse-indoor', 'bar', 'fishpond', 'florist_shop-indoor', 'mansion', 'bus_interior', 'dam', 'patio', 'athletic_field-outdoor', 'grotto', 'basement', 'shopfront', 'carrousel', 'archaelogical_excavation', 'gazebo-exterior', 'beer_hall', 'rainforest', 'windmill', 'kasbah', 'office', 'mountain_path', 'schoolhouse', 'palace', 'bedchamber', 'crevasse', 'canyon', 'plaza', 'archive', 'laundromat', 'diner-outdoor', 'booth-indoor', 'natural_history_museum', 'pagoda', 'volleyball_court-outdoor', 'beer_garden', 'wheat_field', 'rice_paddy', 'art_studio', 'bedroom', 'banquet_hall', 'repair_shop', 'raft', 'glacier', 'army_base', 'playroom', 'football_field', 'subway_station-platform', 'waiting_room', 'field_road', 'waterfall', 'classroom', 'garage-outdoor', 'utility_room', 'dining_room', 'wave', 'youth_hostel', 'phone_booth', 'art_gallery', 'corridor', 'museum-outdoor', 'temple-asia', 'nursing_home', 'snowfield', 'bakery-shop', 'lobby', 'office_building', 'bamboo_forest', 'aquarium', 'galley', 'butte', 'bazaar-indoor', 'library-outdoor', 'field-wild', 'amusement_arcade', 'gas_station', 'wind_farm', 'beach', 'sauna', 'ballroom', 'wet_bar', 'food_court', 'greenhouse-outdoor', 'corn_field', 'courthouse', 'racecourse', 'cabin-outdoor', 'courtyard', 'hayfield', 'farm', 'martial_arts_gym', 'excavation', 'cottage', 'medina', 'ticket_booth', 'childs_room', 'clothing_store', 'zen_garden', 'candy_store', 'fire_escape', 'doorway-outdoor', 'orchestra_pit', 'drugstore', 'mausoleum', 'general_store-outdoor', 'clean_room', 'campsite', 'landing_deck', 'manufactured_home', 'highway', 'playground', 'water_tower', 'orchard', 'underwater-ocean_deep', 'physics_laboratory', 'pavilion', 'art_school', 'tundra', 'river', 'beach_house', 'cemetery', 'baseball_field', 'church-outdoor', 'artists_loft', 'skyscraper', 'museum-indoor', 'ice_shelf', 'landfill', 'science_museum', 'picnic_area', 'soccer_field', 'volcano', 'auto_factory', 'alcove', 'library-indoor', 'hospital', 'junkyard', 'crosswalk', 'pier', 'vegetable_garden', 'closet', 'promenade', 'botanical_garden', 'marsh', 'train_station-platform', 'bullring', 'desert_road', 'auto_showroom', 'trench', 'home_office', 'creek', 'department_store', 'conference_room', 'valley', 'pizzeria', 'oilrig', 'butchers_shop', 'hotel_room', 'igloo', 'yard', 'amphitheater', 'roof_garden', 'house', 'badlands', 'pet_shop', 'railroad_track', 'reception', 'japanese_garden', 'castle', 'campus', 'television_room', 'bathroom', 'field-cultivated', 'lawn', 'hot_spring', 'burial_chamber', 'ice_cream_parlor', 'berth', 'restaurant', 'engine_room', 'dressing_room', 'sushi_bar', 'fountain', 'runway', 'toyshop', 'catacomb', 'construction_site', 'bowling_alley', 'canal-natural', 'topiary_garden', 'formal_garden', 'fabric_store', 'storage_room', 'general_store-indoor', 'airfield', 'park', 'vineyard', 'bookstore', 'barn', 'ski_resort', 'porch', 'jewelry_shop', 'arch', 'golf_course', 'viaduct', 'boardwalk', 'garage-indoor', 'industrial_area', 'bus_station-indoor', 'motel', 'sandbox', 'music_studio', 'dining_hall', 'street', 'elevator-door', 'computer_room', 'kindergarden_classroom', 'coffee_shop', 'television_studio', 'swimming_pool-indoor', 'desert-sand', 'stable', 'assembly_line', 'synagogue-outdoor', 'server_room', 'fastfood_restaurant', 'atrium-public', 'veterinarians_office', 'residential_neighborhood', 'nursery', 'coast', 'driveway', 'operating_room', 'parking_garage-indoor', 'tree_house', 'escalator-indoor', 'pharmacy', 'rock_arch', 'parking_garage-outdoor', 'restaurant_patio', 'conference_center']
BACKGROUND_SIZE_X = 6
BACKGROUND_SIZE_Y = 8

AMBIENT_COLOR = [0.5, 0.5, 0.5]
CAMERA_POSITION = [5, 0, 0]
CAMERA_FOCUS = [-5, 0, 0]
RENDER_SIZE_X = 1024
RENDER_SIZE_Y = 768

def generate_input(image_id, objects_number):

    scene = {}

    scene['global'] = {
        'scene_name': image_id,
        'alter_texture': [
            'skin',
            'hair',
            'pants',
            'shoes',
            'shirt'
        ]
    }

    scene['camera'] = {
        'render_size_x': RENDER_SIZE_X,
        'render_size_y': RENDER_SIZE_Y,
        'position': CAMERA_POSITION
    }

    image_path = get_random_image(BACKGROUND_CLASSES)
    scene['background'] = {
        'path': image_path,
        'position': CAMERA_FOCUS,
        'size_x': BACKGROUND_SIZE_X,
        'size_y': BACKGROUND_SIZE_Y,
        'ambient_color': AMBIENT_COLOR

    }

    scene["lighting"] = {}
    for l in range(random.randint(MIN_LIGHTS,MAX_LIGHTS)):
        if LIGHT_NEGATIVE:
            negative_light = random.choice([True, False])
        else:
            negative_light = False
        scene['lighting']['l' + str(l)] = {
            "light_type": random.choice(LIGHT_TYPES),
            "energy": random.uniform(LIGHT_ENERGY_MIN,LIGHT_ENERGY_MAX),
            "color": [
                random.uniform(LIGHT_COLOR_MIN, 1),
                random.uniform(LIGHT_COLOR_MIN, 1),
                random.uniform(LIGHT_COLOR_MIN, 1)
            ],
            "shadow": LIGHT_CAST_SHADOWS,
            "negative_light": negative_light,
            "position": [
                random.uniform(-LIGHT_POSITION_VAR,LIGHT_POSITION_VAR),
                random.uniform(-LIGHT_POSITION_VAR,LIGHT_POSITION_VAR),
                random.uniform(-LIGHT_POSITION_VAR,LIGHT_POSITION_VAR)
            ],
            "rotation": [
                random.uniform(math.pi, -math.pi),
                random.uniform(math.pi, -math.pi),
                random.uniform(math.pi, -math.pi)
            ]
        }
    
    scene['objects'] = {}
    
    if objects_number == 0:
        return scene
    

    base_size = math.sqrt(1+abs(objects_number))*random.uniform(2.5,5)

    zx_tilt = random.uniform(0.2, 0.7)
    zy_tilt = random.uniform(-0.5, 0.5)
    centers=random.randint(1 + objects_number//20, 2 + objects_number//8)
    cluster_std=1
    x_scale = 3
    y_scale = 15
    z_offset = random.uniform(0.2, 0.4)
    y_offset = 4

    X,_ = make_blobs(cluster_std=cluster_std,random_state=20,n_samples=objects_number,centers=centers, n_features=2)

    for o, xy in enumerate(X):
        rand_object_path = get_lowpoly_human()

        # scale_x = random.uniform(1,8)/base_size
        # scale_y = random.uniform(1,8)/base_size
        # scale_z = random.uniform(1,8)/base_size
        scale = random.uniform(1,8)/base_size

        x = xy[0]/x_scale
        y = ((xy[1]/y_scale) * (-x + y_offset)) * random.choice([-1,1])
        z = -x * zx_tilt + y * zy_tilt + z_offset

        scene['objects']['o' + str(o)] = {
            "path": rand_object_path,
            "2d_vertex": [
                0,
                89,
                115,
                165,
                417,
                459
            ],
            "initial_position": [
                x,
                y,
                z
            ],
            "initial_rotation": [
                random.uniform(0.9 * math.pi/2, 1.1 * math.pi/2),
                0,
                random.uniform(-math.pi, math.pi)
            ],
            'scale': [
                scale,
                scale,
                scale
            ]
        }
        
        scene['objects']['o' + str(o)]['transforms'] = []
        for _ in range(random.randint(OBJECT_TRANSFORMS_MIN,OBJECT_TRANSFORMS_MAX)):
            rand_transform = random.choice(OBJECT_TRANSFORMS)
            if rand_transform == 'CAST':
                transform = {
                    "type": "CAST",
                    "factor": random.uniform(OBJECT_TRANSFORMS_CAST_FACTOR_MIN, OBJECT_TRANSFORMS_CAST_FACTOR_MAX),
                    "cast_type": random.choice(OBJECT_TRANSFORMS_CAST_TYPES)
                }
            elif rand_transform == 'SOLIDIFY':
                transform = {
                    "type": "SOLIDIFY",
                    "thickness": random.uniform(OBJECT_TRANSFORMS_SOLIDIFY_THICKNESS_MIN, OBJECT_TRANSFORMS_SOLIDIFY_THICKNESS_MAX)
                }
            elif rand_transform == 'SIMPLE_DEFORM':
                transform = {
                    "type": "SIMPLE_DEFORM",
                    "angle": random.uniform(OBJECT_TRANSFORMS_SIMPLE_DEFORM_ANGLE_MIN, OBJECT_TRANSFORMS_SIMPLE_DEFORM_ANGLE_MAX)
                }
            elif rand_transform == 'RANDOMIZE':
                transform = {
                    "type": "RANDOMIZE",
                    "amount": random.uniform(OBJECT_TRANSFORMS_RANDOMIZE_FACTOR_MIN, OBJECT_TRANSFORMS_RANDOMIZE_FACTOR_MAX)
                }
            scene['objects']['o' + str(o)]['transforms'].append(transform)

    return scene


