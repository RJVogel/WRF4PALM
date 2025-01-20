#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import xarray as xr
from tqdm import tqdm
from functools import partial
from multiprocess import Pool
def zinterp( ds, var,lvl ):
    data = ds.salem.wrf_zlevel(var, levels=lvl, use_multiprocessing=False)
    return (lvl, data)

def multi_zinterp(max_pool, ds_in, var, zcoord, ds_out):
    with Pool(max_pool) as p:
        pool_outputs = list(
            tqdm(
                p.imap(partial(zinterp, ds_in, var),zcoord), total=len(zcoord),
                position=0, leave=True
            )
        )
    p.join()
    ## convert dictionary back to dataset
    pool_dict = dict(pool_outputs)
    for lvl in zcoord:
        if var == "W":
            # Create new DataArray with PALM dimensions
            data = xr.DataArray(
                pool_dict[lvl],
                dims=['time', 'y', 'x'],
                coords={
                    'time': ds_out.time,
                    'y': ds_out.y,
                    'x': ds_out.x,
                    'zw': lvl
                }
            )
            ds_out[var].loc[dict(time=ds_out.time, y=ds_out.y, x=ds_out.x, zw=lvl)] = data
        else:
            # Create new DataArray with PALM dimensions
            # Handle staggered grid for U component
            if var == "U":
                data = xr.DataArray(
                    pool_dict[lvl],
                    dims=['time', 'y', 'xu'],
                    coords={
                        'time': ds_out.time,
                        'y': ds_out.y,
                        'xu': ds_out.xu,
                        'z': lvl
                    }
                )
                ds_out[var].loc[dict(time=ds_out.time, y=ds_out.y, xu=ds_out.xu, z=lvl)] = data
            elif var == "V":
                # Handle staggered grid for V component
                data = xr.DataArray(
                    pool_dict[lvl],
                    dims=['time', 'yv', 'x'],
                    coords={
                        'time': ds_out.time,
                        'yv': ds_out.yv,
                        'x': ds_out.x,
                        'z': lvl
                    }
                )
                ds_out[var].loc[dict(time=ds_out.time, yv=ds_out.yv, x=ds_out.x, z=lvl)] = data
            else:
                # Regular grid for other variables
                data = xr.DataArray(
                    pool_dict[lvl],
                    dims=['time', 'y', 'x'],
                    coords={
                        'time': ds_out.time,
                        'y': ds_out.y,
                        'x': ds_out.x,
                        'z': lvl
                    }
                )
                ds_out[var].loc[dict(time=ds_out.time, y=ds_out.y, x=ds_out.x, z=lvl)] = data
    return ds_out[var]

