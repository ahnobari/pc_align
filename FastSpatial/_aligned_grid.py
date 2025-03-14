import numpy as np

def generate_full_rotation_grid(num_angles=200):
    """
    Vectorized generation of the full grid of rotation matrices.
    
    The procedure is:
      1. Generate 18 base rotations that align one chosen source axis
         with one chosen target axis (with ± ambiguity).
      2. For each base rotation, generate num_angles rotations about
         the aligned (source) axis.
    
    Returns:
        A NumPy array of shape (18*num_angles, 3, 3).
    """
   
    e = np.eye(3)
    base_list = []
    aligned_axes = []
    
    # Build the 18 base alignments
    for i in range(3):
        for j in range(3):
            for s in [1, -1]:
                R = np.zeros((3, 3))
                R[:, i] = s * e[j]
                
                remaining_source = [k for k in range(3) if k != i]
                remaining_target = [k for k in range(3) if k != j]
                R[:, remaining_source[0]] = e[remaining_target[0]]
                R[:, remaining_source[1]] = e[remaining_target[1]]
                
                if np.linalg.det(R) < 0:
                    R[:, remaining_source[0]] = -R[:, remaining_source[0]]
                base_list.append(R)
                aligned_axes.append(i)
    base_array = np.stack(base_list, axis=0)
    aligned_axes = np.array(aligned_axes)
    
    
    angles = np.linspace(0, 2*np.pi, num=num_angles, endpoint=False)
    cos_theta = np.cos(angles)
    sin_theta = np.sin(angles)

    R_extras = {}
    
    R_extras[0] = np.empty((num_angles, 3, 3))
    R_extras[0][:, 0, 0] = 1
    R_extras[0][:, 0, 1] = 0
    R_extras[0][:, 0, 2] = 0
    R_extras[0][:, 1, 0] = 0
    R_extras[0][:, 1, 1] = cos_theta
    R_extras[0][:, 1, 2] = -sin_theta
    R_extras[0][:, 2, 0] = 0
    R_extras[0][:, 2, 1] = sin_theta
    R_extras[0][:, 2, 2] = cos_theta
    
    R_extras[1] = np.empty((num_angles, 3, 3))
    R_extras[1][:, 0, 0] = cos_theta
    R_extras[1][:, 0, 1] = 0
    R_extras[1][:, 0, 2] = sin_theta
    R_extras[1][:, 1, 0] = 0
    R_extras[1][:, 1, 1] = 1
    R_extras[1][:, 1, 2] = 0
    R_extras[1][:, 2, 0] = -sin_theta
    R_extras[1][:, 2, 1] = 0
    R_extras[1][:, 2, 2] = cos_theta
    
    R_extras[2] = np.empty((num_angles, 3, 3))
    R_extras[2][:, 0, 0] = cos_theta
    R_extras[2][:, 0, 1] = -sin_theta
    R_extras[2][:, 0, 2] = 0
    R_extras[2][:, 1, 0] = sin_theta
    R_extras[2][:, 1, 1] = cos_theta
    R_extras[2][:, 1, 2] = 0
    R_extras[2][:, 2, 0] = 0
    R_extras[2][:, 2, 1] = 0
    R_extras[2][:, 2, 2] = 1

    R_all = np.empty((base_array.shape[0], num_angles, 3, 3))
    for axis in [0, 1, 2]:
        inds = np.where(aligned_axes == axis)[0]
        if len(inds) > 0:
            base_group = base_array[inds]
            extra_group = R_extras[axis]
            R_all[inds] = np.matmul(base_group[:, None, :, :], extra_group[None, :, :, :])
    
    R_all = R_all.reshape(-1, 3, 3)
    return R_all