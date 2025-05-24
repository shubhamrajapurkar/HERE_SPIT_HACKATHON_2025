import fiona

gpkg_file = "faulty_pois.gpkg" # Make sure this path is correct

try:
    layer_names = fiona.listlayers(gpkg_file)
    print(f"Layers found in '{gpkg_file}':")
    for name in layer_names:
        print(f"  - '{name}'") # Print with single quotes for easy copying
except Exception as e:
    print(f"Error accessing GeoPackage '{gpkg_file}': {e}")
    print("Please ensure the file path is correct and the file is a valid GeoPackage.")