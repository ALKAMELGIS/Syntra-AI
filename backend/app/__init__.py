# Point PROJ/GDAL at this environment's data before anything imports rasterio or
# pyproj (a PostGIS/QGIS install can set PROJ_LIB system-wide; see core/proj_env).
from app.core.proj_env import ensure_own_geodata as _ensure_own_geodata

_ensure_own_geodata()
