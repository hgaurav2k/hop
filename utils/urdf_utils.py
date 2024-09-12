import numpy as np 
def read_xml(filename):
    import xml.etree.ElementTree as Et
    root = Et.parse(filename).getroot()
    return root


def get_link_meshes_from_urdf(urdf_file,link_names):
    root = read_xml(urdf_file)
    link_meshfiles =[]
    for link_name in link_names:
        for link in root.findall('link'):
            if link.attrib['name'] == link_name:
                for mesh in link.findall('visual/geometry/mesh'):
                    link_meshfiles.append(mesh.attrib['filename'])

    assert len(link_meshfiles) == len(link_names)   
    return link_meshfiles


def load_asset_files_public(asset_root):
    import os 
    folder_name = 'pybullet-URDF-models/urdf_models/models'
    asset_files = {}

    for root, dirs, files in os.walk(os.path.join(asset_root,folder_name)):
        
        for file in files:
            if file.endswith("model.urdf"):
                obj_name = root.split('/')[-1]
                dir  = root[len(asset_root)+1:]
                asset_files[obj_name]=os.path.join(dir, file)
    
    return asset_files




def load_asset_files_ycb(asset_root,folder_name='ycb_real_inertia'):

    import os 
    asset_files = {}

    for root, dirs, files in os.walk(os.path.join(asset_root,folder_name)):

        for file in files:
            if file.endswith(".urdf"):
                obj_name = file.split('.')[0]
                dir  = root[len(asset_root)+1:]
                asset_files[obj_name]={}    
                asset_files[obj_name]['urdf']=os.path.join(dir, file)
                asset_files[obj_name]['mesh']=os.path.join(dir, file.split('.')[0]+'/google_16k/textured.obj')
                assert os.path.exists(os.path.join(asset_root,asset_files[obj_name]['mesh']))
                assert os.path.exists(os.path.join(asset_root,asset_files[obj_name]['urdf']))
                    
    return asset_files

def load_asset_files_ycb_lowmem(asset_root,folder_name='ycb_real_inertia'):
    import os 
    asset_files = {}

    for root, dirs, files in os.walk(os.path.join(asset_root,folder_name)):

        for file in files:
            if file.endswith(".urdf"):
                obj_name = file.split('.')[0]
                number = obj_name.split('_')[0]
                print(obj_name,number)
                if number in ['070-a','070-b','072','036','032','029','048','027','019','032','026']:
                    dir  = root[len(asset_root)+1:]
                    asset_files[obj_name]={}    
                    asset_files[obj_name]['urdf']=os.path.join(dir, file)
                    asset_files[obj_name]['mesh']=os.path.join(dir, file.split('.')[0]+'/google_16k/textured.obj')
                    assert os.path.exists(os.path.join(asset_root,asset_files[obj_name]['mesh']))
                    assert os.path.exists(os.path.join(asset_root,asset_files[obj_name]['urdf'])) 
                    
    return asset_files


def fix_ycb_scale(asset_root):
    import os 
    import shutil 
    import xml.etree.ElementTree as Et
    folder_name = 'ycb'
    new_folder_name = 'ycb_scaled'
    if not os.path.exists(os.path.join(asset_root,new_folder_name)):
        shutil.copytree(os.path.join(asset_root,folder_name), os.path.join(asset_root,new_folder_name))

    for root, dirs, files in os.walk(os.path.join(asset_root,new_folder_name)):
        for file in files:
            if file.endswith(".urdf"):
                filepath = os.path.join(root, file)
                urdf = read_xml(filepath)
                for mesh in urdf.findall(f'.//collision/geometry/'):
                    mesh.attrib['scale']='1 1 1'
                for mesh in urdf.findall(f'.//visual/geometry/'):
                    mesh.attrib['scale']='1 1 1'

                new_xml = Et.ElementTree()
                new_xml._setroot(urdf)
                with open(filepath, "wb") as f:
                    new_xml.write(f)

    return





def get_vol_ratio(scale1,scale2):
    nums1 = [float(s) for s in scale1.split(' ')]
    nums2 = [float(s) for s in scale2.split(' ')]
    nums1 = np.array(nums1)
    nums2 = np.array(nums2)
    return np.prod(nums1)/np.prod(nums2)
