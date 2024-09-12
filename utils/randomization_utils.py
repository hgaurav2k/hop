import numpy as np 
from termcolor import cprint


##CAN ONLY BE DONE AT THE START OF SIMULATION##
def randomize_table_z(gym,env_ptr,table_handle,table_rand_config):
    #does not work. need to change table position differently.
    fr_z = np.random.uniform(table_rand_config['lower'],table_rand_config['upper'])
    prop = gym.get_actor_rigid_body_properties(env_ptr, table_handle)
    assert len(prop) == 1
    print(fr_z)
    obj_com = prop[0].com.z*fr_z
    prop[0].com.z = obj_com
    gym.set_actor_rigid_body_properties(env_ptr, table_handle, prop)


##CAN ONLY BE DONE AT THE START OF SIMULATION##
def randomize_object_scale(gym,env_ptr,object_handle,object_rand_config):

    scale = np.random.uniform(object_rand_config['lower'], object_rand_config['upper'])
    gym.set_actor_scale(env_ptr, object_handle,scale)
    return scale 


##CAN ONLY BE DONE AT THE START OF SIMULATION##
def randomize_object_mass(gym,env_ptr,object_handle,objmass_rand_config):

    prop = gym.get_actor_rigid_body_properties(env_ptr, object_handle)
    ret = []
    for p in prop:
        fr = np.random.uniform(objmass_rand_config['lower'], objmass_rand_config['upper'])
        p.mass = p.mass*fr 
        p.inertia.x = p.inertia.x*fr
        p.inertia.y = p.inertia.y*fr
        p.inertia.z = p.inertia.z*fr
        ret.append(p.mass)
        
    gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)

    return ret 


##CAN ONLY BE DONE AT THE START OF SIMULATION##
def randomize_friction(gym,env_ptr,handle,rand_friction_config):
    
    rand_friction = np.random.uniform(rand_friction_config['lower'], rand_friction_config['upper'])
    rest = np.random.uniform(rand_friction_config['lower'], rand_friction_config['upper'])
    props = gym.get_actor_rigid_shape_properties(env_ptr, handle)
    friction = []
    restitution = []
    for p in props:
        p.friction = rand_friction*p.friction
        p.restitution = rest*p.restitution
        friction.append(p.friction)
        restitution.append(p.restitution)
    
    gym.set_actor_rigid_shape_properties(env_ptr, handle, props)

    return friction,restitution 

# def randomize_friction(gym,env_ptr,hand_handle,object_handle,rand_friction_config):

#     rand_friction = np.random.uniform(rand_friction_config['lower'], rand_friction_config['upper'])
#     obj_restitution = np.random.uniform(rand_friction_config['lower'], rand_friction_config['upper'])
#     hand_props = gym.get_actor_rigid_shape_properties(env_ptr, hand_handle)
#     hand_friction = []
#     hand_restitution = []
#     for p in hand_props:
#         p.friction = rand_friction
#         p.restitution = obj_restitution
#         hand_friction.append(p.friction)
#         hand_restitution.append(p.restitution)
    
#     gym.set_actor_rigid_shape_properties(env_ptr, hand_handle, hand_props)


#     rand_friction = np.random.uniform(rand_friction_config['lower'], rand_friction_config['upper'])
#     obj_rest = np.random.uniform(rand_friction_config['lower'], rand_friction_config['upper'])
#     obj_friction  = []
#     obj_restitution = []
#     obj_props = gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
#     for p in obj_props:
#         p.friction = rand_friction*p.friction
#         p.restitution = obj_rest*p.restitution
#         obj_friction.append(p.friction)
#         obj_restitution.append(p.restitution)
    
#     gym.set_actor_rigid_shape_properties(env_ptr, object_handle, obj_props)

#     return hand_friction, hand_restitution, obj_friction, obj_restitution #not sure if just one value can influence the full policy but okay for now. 


# def randomize_object_position(env):
#     "already randomized in code"
#     pass 

# def randomize_robot_damping(env):
#     pass 

# def randomize_robot_stiffness(env):
#     pass 

