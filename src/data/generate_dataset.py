import time
import os
import subprocess
import json
import ast
import requests
from threading import Thread    


def dataset(args):

    if args.category.lower() == 'people':
        # from src.data.people import generate_input
        from src.data.people_trans_rand import generate_input
        # from src.data.people_trans_extrude import generate_input
    if args.category.lower() == 'penguins':
        # from src.data.penguins import generate_input
        # from src.data.penguins_trans_rand import generate_input
        from src.data.penguins_trans_extrude import generate_input
    if args.category.lower() == 'vehicles':
        # from src.data.vehicles import generate_input
        # from src.data.vehicles_trans_scale import generate_input
        # from src.data.vehicles_no_trans import generate_input
        # from src.data.vehicles_trans_rand import generate_input
        from src.data.vehicles_trans_extrude import generate_input
        
    if args.category.lower() == 'apples':
        from src.data.apples import generate_input

    os.makedirs('/dm_count/data/raw/{}'.format(args.dataset_name), exist_ok=True)

    distribution = [int(args.variations)] * int(args.max)

    image_count = []
    instances = {}

    # Define instances
    for instance in range(1, int(args.workers) + 1):
        instances['docker_blendercam_' + str(instance)] = True

    # Check if blendercam is scaled properly
    for instance in list(instances.keys()):
        command = ['ping', '-c', '1', instance]
        server_active =  subprocess.call(command, stdout=open(os.devnull, 'wb')) == 0
        assert server_active == True, 'Please check that blendercam has enough instances running'

    # Convert distribution to list
    distribution_list = []
    for i, d in enumerate(distribution):
        for x in range(0, d):
            distribution_list.append(i)
    
    # import pdb;pdb.set_trace()

    # Threaded function that send jobs to the instances and stores the response
    def send_job(instance, scene, port=5000):
        try:
            response = requests.get('http://{}:{}/'.format(instance, 5000),
                        json=scene)

            if 'succeed' in response.headers.get('info'):

                scene['response'] = ast.literal_eval(response.headers.get('info'))
                # import pdb;pdb.set_trace()
                with open('/dm_count/data/raw/{}/{}'.format(args.dataset_name, str(scene['response']['image_name'])), 'wb') as f:
                    f.write(response.content)

                image_count.append(scene['response']['image_name'])
                

                with open('/dm_count/data/raw/{}/{}.json'.format(args.dataset_name, str(len(image_count)).zfill(8)), 'w') as fp:
                    json.dump(scene, fp)
                

                instances[instance] = True
                print('Images generated: {} / {}'.format(str(len(image_count)), len(distribution_list)), end='\r')
            else:
                print('Error, check JSON')
        except Exception as e:
            print(e)
            instances[instance] = True

    while len(image_count) < len(distribution_list):
        # Starts a thread when an instance is available
        for instance, available in instances.items():
            if len(image_count) >= len(distribution_list) : break
            if available:
                
                # Generate scene based on the given distribution
                number_of_objects = distribution_list[len(image_count)]
                
                scene = generate_input(str(len(image_count)).zfill(8), number_of_objects)
                # Create and start thread
                try:
                    thread = Thread( target=send_job, args=(instance, scene, number_of_objects) )
                    thread.start()
                    
                    # Set instance to busy
                    instances[instance] = False
                except Exception as e:
                    print(e)
                    instances[instance] = True
        
        # Sleep while waiting for instances to be available
        time.sleep(0.3)