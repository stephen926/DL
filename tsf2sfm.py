import numpy as np
import spharm

def spharm_transform(data_origin, n_keep = None, truncation_level = 122):
    """
    将数据字典转换为频谱字典
    """
    skip_vars = ['lon', 'lat']
    
    # 检测空间维度
    nlat, nlon = None, None
    for key, value in data_origin.items():
        if key not in skip_vars and value is not None:
            if hasattr(value, 'shape') and len(value.shape) >= 2:
                nlat, nlon = value.shape[-2:]
                break
    
    if nlat is None or nlon is None:
        raise ValueError("无法检测空间维度")
    
    if n_keep is None:
        n_keep = (truncation_level + 1) ** 2
    grid = spharm.Spharmt(nlon, nlat, rsphere=6.371e6, gridtype='regular')
    
    data_train = {}
    
    for key, data in data_origin.items():
        if key in skip_vars or data is None:
            continue
        
        try:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            if data.shape != (nlat, nlon):
                if data.size == nlat * nlon:
                    data = data.reshape(nlat, nlon)
                else:
                    continue
            
            data = np.nan_to_num(data, nan=0.0)
            data = np.clip(data, -1e10, 1e10)
            
            spec = grid.grdtospec(data)
            data_train[key] = spec[:n_keep]
            
        except:
            continue
    
    return data_train

def spectral_to_grid(data_train, nlat=181, nlon=360):
    """
    频谱系数转回网格数据
    """
    grid = spharm.Spharmt(nlon, nlat, rsphere=6.371e6, gridtype='regular')
    full_spec_length = (nlat * (nlat + 1)) // 2
    
    data_reconstructed = {}
    
    for key, spec_coeffs in data_train.items():
        full_spec = np.zeros(full_spec_length, dtype=np.complex128)
        full_spec[:len(spec_coeffs)] = spec_coeffs
        data_reconstructed[key] = grid.spectogrd(full_spec)
    
    return data_reconstructed