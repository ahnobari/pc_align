import numpy as np
import cupy as cp
from typing import Tuple, List, Union
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree

def procrustes(point_clouds: Union[cp.ndarray, np.ndarray]) -> Tuple[Union[cp.ndarray, cp.ndarray], Union[cp.ndarray, cp.ndarray], Union[cp.ndarray, cp.ndarray]]:
    '''
    Normalize point clouds by centering and scaling them according to the Procrustes analysis (Assumes similar sampling and strucure when comparing point clouds).
    
    Parameters:
        point_clouds: Union[cp.ndarray, np.ndarray]
            Tuple of point clouds to be normalized.
        
    Returns:
        point_clouds_normalized: Union[cp.ndarray, cp.ndarray]
            Normalized point clouds.
        centers: Union[cp.ndarray, cp.ndarray]
            Centers of the point clouds.
        scales: Union[cp.ndarray, cp.ndarray]
            Scales of the point clouds.
    '''

    centers = point_clouds.mean(axis=1)
    
    if isinstance(point_clouds, np.ndarray):
        scales = np.sqrt(np.mean(np.square(point_clouds - centers[:,None]), axis=(1,2)))
    else:
        scales = cp.sqrt(cp.mean(cp.square(point_clouds - centers[:,None]), axis=(1,2)))
    
    point_clouds_normalized = (point_clouds - centers[:,None]) / scales[:,None]
    
    return point_clouds_normalized, centers, scales
    
def principal_axes(point_clouds: Union[cp.ndarray, cp.ndarray], normalize: bool = False) -> Union[cp.ndarray, cp.ndarray]:
    '''
    Compute the principal axes of the point clouds.
    
    Parameters:
        point_clouds: Union[cp.ndarray, np.ndarray]
            Tuple of point clouds to be normalized.
        normalize: bool
            Normalize the point clouds before computing the principal axes.
    
    Returns:
        principal_axes: Union[cp.ndarray, cp.ndarray]
            Principal axes of the point clouds.
    '''
    
    if normalize:
        point_clouds_normalized, centers, scales = procrustes(point_clouds)
    else:
        point_clouds_normalized = point_clouds
        
    cov = point_clouds_normalized.transpose(0, 2, 1) @ point_clouds_normalized
    
    if isinstance(point_clouds, np.ndarray):
        _, principal_axes = np.linalg.eigh(cov)
        
    else:
        _, principal_axes = cp.linalg.eigh(cov)
        
    return principal_axes.transpose(0, 2, 1)

def align_clouds(source: Union[cp.ndarray, np.ndarray], target: Union[cp.ndarray, np.ndarray], normalize: bool = False) -> Union[cp.ndarray, cp.ndarray]:
    '''
    Align source point clouds to target point clouds.
    
    Parameters:
        source: Union[cp.ndarray, np.ndarray]
            Source point cloud.
        target: Union[cp.ndarray, np.ndarray]
            Target point cloud.
        normalize: bool
            Normalize the point clouds before computing the principal axes.
    
    Returns:
        aligned_source: Union[cp.ndarray, cp.ndarray]
            Aligned source point cloud.
        normalized_chamfer_distance: float
            Normalized Chamfer distance between the source and target point clouds after alignment.
    '''
    
    if normalize:
        source_normalized, centers, scales = procrustes(source[None])
        target_normalized, centers_t, scales_t = procrustes(target[None])
        source_normalized = source_normalized[0]
        target_normalized = target_normalized[0]
        centers_t = centers_t[0]
        scales_t = scales_t[0]
    else:
        source_normalized = source
        target_normalized = target
        
    principal_axes_source = principal_axes(source_normalized[None], normalize=False)[0]
    principal_axes_target = principal_axes(target_normalized[None], normalize=False)[0]
    
    # go through all 8 possible alignments of the principal axes
    best_alignment = None
    best_error = np.inf
    
    target_tree = KDTree(target_normalized)
    
    for i in range(8):
        alignment = np.array([i & 1, (i >> 1) & 1, (i >> 2) & 1])*2 - 1
        flipped_axes = principal_axes_source * alignment[:,None]
        
        R = principal_axes_target.T @ flipped_axes
        
        aligned_source = source_normalized @ R.T
        
        source_tree = KDTree(aligned_source)
        
        cd = source_tree.query(target_normalized)[0].mean() + target_tree.query(aligned_source)[0].mean()
        
        if cd < best_error:
            best_error = cd
            best_alignment = alignment
            
    R = principal_axes_target.T @ (principal_axes_source * best_alignment[:,None])
    aligned_source = source_normalized @ R.T
    
    if normalize:
        aligned_source = aligned_source * scales_t + centers_t
        
    return aligned_source, best_error
        