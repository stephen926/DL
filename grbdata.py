import numpy as np
from scipy.interpolate import interp1d

def zoom(x):
    # tan h 
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# 池化降尺度
# def downsample(data_dict, skip_keys=('lon', 'lat')):
#     pooled_dict = {}

#     for key, value in data_dict.items():
#         if key in skip_keys:
#             pooled_dict[key] = value
#             continue

#         if value.shape != (361, 720):
#             raise ValueError(f"{key} shape {value.shape} ≠ (361, 720)")

#         # 1. 纬度方向线性插值：361->181
#         lat_src = np.linspace(-90, 90, 361)
#         lat_dst = np.linspace(-90, 90, 181)
#         interp_func = interp1d(lat_src, value, axis=0, kind='linear')
#         value_interp = interp_func(lat_dst)  # shape: (181, 720)

#         # 2. 复制两行极点附近数据，但对复制的这两行经度方向移动180°（360列）
#         lat_src_fine = np.linspace(-90, 90, 361)
#         idx_north_89_5 = np.abs(lat_src_fine - 89.5).argmin()
#         idx_south_89_5 = np.abs(lat_src_fine + 89.5).argmin()

#         north_row = value[idx_north_89_5, :]
#         south_row = value[idx_south_89_5, :]

#         # 只对这两行做经度方向180°移动
#         north_row_shifted = np.roll(north_row, shift=360)
#         south_row_shifted = np.roll(south_row, shift=360)

#         # 3. 拼接：南极复制行 + 插值后数据 + 北极复制行
#         value_extended = np.vstack([
#             south_row_shifted[np.newaxis, :],
#             value_interp,
#             north_row_shifted[np.newaxis, :]
#         ])  # shape: (183, 720)

#         # 4. 经度方向池化2列均值 (183, 360)
#         pooled = value_extended.reshape(183, 360, 2).mean(axis=2)

#         # 5. 纬度方向线性插值回181行
#         lat_extended_src = np.linspace(-90 - (90 - 89.5), 90 + (90 - 89.5), 183)
#         interp_func_lat = interp1d(lat_extended_src, pooled, axis=0, kind='linear')
#         pooled_final = interp_func_lat(lat_dst)  # shape: (181, 360)

#         pooled_dict[key] = pooled_final

#     return pooled_dict

from scipy.interpolate import RegularGridInterpolator
# 插值降尺度
def downsample(data_dict, skip_keys=('lon', 'lat')):
    pooled_dict = {}

    # 原始纬度和经度坐标
    lat_src = np.linspace(-90, 90, 361)
    lon_src = np.linspace(0, 360, 720, endpoint=False)  # 0~360开区间，保持经度不重叠

    # 目标纬度和经度
    lat_dst = np.linspace(-90, 90, 181)
    lon_dst = np.linspace(0, 360, 360, endpoint=False)

    # 为插值做准备，生成网格点
    pts_src = (lat_src, lon_src)

    for key, value in data_dict.items():
        if key in skip_keys:
            pooled_dict[key] = value
            continue

        if value.shape != (361, 720):
            raise ValueError(f"{key} shape {value.shape} ≠ (361, 720)")

        # 创建插值函数
        interp_func = RegularGridInterpolator(pts_src, value, method='linear', bounds_error=False, fill_value=None)

        # 生成目标网格点对 (lat, lon) 的组合，注意经度作为第二维
        lat_grid, lon_grid = np.meshgrid(lat_dst, lon_dst, indexing='ij')
        pts_dst = np.array([lat_grid.flatten(), lon_grid.flatten()]).T  # (181*360, 2)

        # 插值并reshape回目标形状
        value_interp = interp_func(pts_dst).reshape(181, 360)

        pooled_dict[key] = value_interp

    return pooled_dict

def grbdata(ctrl_00_filepath, ctrl_06_filepath, spread_filepath, layer, zoomin = True, ds = True):
    import cfgrib
    import metpy.calc as mpcalc
    from metpy.units import units
    from windspharm.xarray import VectorWind

    ds_train_00 = cfgrib.open_dataset(ctrl_00_filepath, filter_by_keys={'typeOfLevel': 'isobaricInhPa','level': layer})
    ds_train_06 = cfgrib.open_dataset(ctrl_06_filepath, filter_by_keys={'typeOfLevel': 'isobaricInhPa','level': layer})
    ds_spr = cfgrib.open_dataset(spread_filepath, filter_by_keys={'typeOfLevel': 'isobaricInhPa','level': layer})

    lon,lat = ds_train_00.longitude,ds_train_00.latitude
    x, y = np.meshgrid(lon, lat)

    gh_ctrl_00 = ds_train_00.gh
    t_ctrl_00 = ds_train_00.t
    r_ctrl_00 = ds_train_00.r
    u_ctrl_00 = ds_train_00.u
    v_ctrl_00 = ds_train_00.v

    gh_ctrl = ds_train_06.gh
    t_ctrl = ds_train_06.t
    r_ctrl = ds_train_06.r
    u_ctrl = ds_train_06.u
    v_ctrl = ds_train_06.v

    gh_diff = gh_ctrl - gh_ctrl_00
    t_diff = t_ctrl - t_ctrl_00
    r_diff = r_ctrl - r_ctrl_00
    u_diff = u_ctrl - u_ctrl_00
    v_diff = v_ctrl - v_ctrl_00

    def grad(var):
        x_delta = 2 * units.km
        y_delta = 1 * units.km
        grd = mpcalc.gradient(var, 
                            deltas=(y_delta, x_delta),
                            #   coordinates=(lat, lon),
                            )
        grdx = grd[0]
        grdy = grd[1]
        grds = np.sqrt(grdx**2+grdy**2)
        return grds

    gh_grad = grad(gh_ctrl).magnitude
    t_grad = grad(t_ctrl).magnitude
    r_grad = grad(r_ctrl).magnitude
    u_grad = grad(u_ctrl).magnitude
    v_grad = grad(v_ctrl).magnitude

    u_fixed = u_ctrl.rename({'latitude': 'lat', 'longitude': 'lon'})
    v_fixed = v_ctrl.rename({'latitude': 'lat', 'longitude': 'lon'})

    # 确保纬度是从南到北排列
    u_fixed = u_fixed.sortby('lat')
    v_fixed = v_fixed.sortby('lat')

    w = VectorWind(u_fixed, v_fixed)
    div_ctrl = w.divergence() * 1e4
    vor_ctrl = w.vorticity() * 1e4

    if zoomin == True:
        div_ctrl = zoom(div_ctrl)
        vor_ctrl = zoom(vor_ctrl) 

    gh_spr = ds_spr.gh
    t_spr = ds_spr.t
    r_spr = ds_spr.r
    u_spr = ds_spr.u
    v_spr = ds_spr.v

    data_train = {
    'gh': gh_ctrl.values,
    't': t_ctrl.values,
    'r': r_ctrl.values,
    'u': u_ctrl.values,
    'v': v_ctrl.values,
    'gh_diff': gh_diff.values,
    't_diff': t_diff.values,
    'r_diff': r_diff.values,
    'u_diff': u_diff.values,
    'v_diff': v_diff.values,
    'gh_grad': gh_grad,
    't_grad': t_grad,
    'r_grad': r_grad,
    'u_grad': u_grad,
    'v_grad': v_grad,
    'div_ctrl': div_ctrl.values,
    'vor_ctrl': vor_ctrl.values,
    'lon':x,
    'lat':y
    }
    
    data_spread = {
    'gh': gh_spr.values,
    't': t_spr.values,
    'r': r_spr.values,
    'u': u_spr.values,
    'v': v_spr.values,
    }
    
    if ds == True:
        lon_lr = np.linspace(0, 359, 360)
        lat_lr = np.linspace(90, -90, 181)
        x, y = np.meshgrid(lon_lr, lat_lr)

        data_train['lon'] = x
        data_train['lat'] = y
 
        data_train = downsample(data_train)
        data_spread = downsample(data_spread)

    return data_train, data_spread

def resplot(var, variablename, titlename, cmap, extend, save=False, savepath = 'D://DL//train//'):
    from matplotlib import pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.mpl.ticker as cticker

    fig = plt.figure(figsize=(12, 8))
    proj = ccrs.PlateCarree(central_longitude = 180)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.6], projection=proj)
    ax.set_global()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    lon, lat = np.linspace(0, 360, var.shape[1]), np.linspace(-90, 90, var.shape[0])
    x, y = np.meshgrid(lon - 180, lat)

    if variablename == 'gh':
        levels = np.arange(0,35,5)
    elif variablename == 't':
        levels = np.arange(0,3.5,0.5)
    elif variablename == 'r':
        levels = np.arange(0,40,5)
    elif variablename == 'u' or 'v':
        levels = np.arange(0,11,2)
    elif variablename == 'div' or 'vor':
        levels = np.arange(-1,1.1,0.1)

    if titlename == 'Difference':
       levels = None

    c = ax.contourf(x, y, var, levels = levels, cmap=cmap, extend=extend, transform=ccrs.PlateCarree())

    plt.colorbar(c, ax=ax, orientation="vertical", shrink=0.75)
    plt.title(f"{variablename} {titlename}")
    if save:
        plt.savefig(f"{savepath}{variablename}_{titlename}.png", bbox_inches='tight')
    plt.show()