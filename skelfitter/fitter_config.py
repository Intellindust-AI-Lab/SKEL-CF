import os 

# root path
package_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

skel_folder = os.path.join(package_directory, 'data_inputs/body_models/skel/')
smpl_folder = os.path.join(package_directory, 'data_inputs/body_models/SMPL/')
fitting_mask_file = os.path.join(package_directory, 'body_models/skel/alignment/riggid_parts_mask.pkl')
default_config_file = os.path.join(package_directory, 'alignment/default_config.yaml')

print(skel_folder)