#!/usr/bin/env python3
"""
Check available block groups in the shapefile
"""

import geopandas as gpd
import os

def check_available_blocks():
    """Check what block groups are available in the shapefile"""
    
    bgfile_path = "data/2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp"
    
    if not os.path.exists(bgfile_path):
        print(f"ERROR: Shapefile not found: {bgfile_path}")
        return
    
    try:
        print("Reading shapefile...")
        gdf = gpd.read_file(bgfile_path)
        print(f"✓ Shapefile loaded with {len(gdf)} block groups")
        
        # Create short GEOID column
        gdf["short_GEOID"] = gdf["GEOID"].str[-7:]
        
        print(f"\nFirst 10 GEOIDs:")
        for i, geoid in enumerate(gdf["GEOID"].head(10)):
            short_geoid = geoid[-7:]
            print(f"  {geoid} -> {short_geoid}")
        
        print(f"\nAll short GEOIDs (first 30):")
        short_geoids = gdf["short_GEOID"].tolist()
        for i, short_geoid in enumerate(short_geoids[:30]):
            print(f"  {short_geoid}")
            
        print(f"\nTotal available short GEOIDs: {len(short_geoids)}")
        
        # Check for patterns
        print(f"\nGEOID patterns:")
        prefixes = set([geoid[:3] for geoid in short_geoids])
        print(f"  Prefixes: {sorted(prefixes)}")
        
        # Check the requested GEOIDs
        requested_geoids = [
            "1500000US420034980001", "1500000US420034980002", "1500000US420034993001", 
            "1500000US420034993002", "1500000US420034994001", "1500000US420034994002",
            "1500000US420034994003", "1500000US420034995001", "1500000US420034995002",
            "1500000US420034995003", "1500000US420034996001", "1500000US420034996002",
            "1500000US420034996003", "1500000US420034997001", "1500000US420034997002",
            "1500000US420034997003", "1500000US420034998001", "1500000US420034998002",
            "1500000US420034998003", "1500000US420034999001", "1500000US420034999002",
            "1500000US420034999003", "1500000US420035001001", "1500000US420035001002",
            "1500000US420035001003", "1500000US420035002001", "1500000US420035002002",
            "1500000US420035003001", "1500000US420035003002", "1500000US420035003003"
        ]
        
        requested_short = [geoid[-7:] for geoid in requested_geoids]
        print(f"\nRequested short GEOIDs:")
        for short_geoid in requested_short:
            print(f"  {short_geoid}")
        
        print(f"\nChecking availability:")
        available = []
        missing = []
        for short_geoid in requested_short:
            if short_geoid in short_geoids:
                available.append(short_geoid)
            else:
                missing.append(short_geoid)
        
        print(f"\nAvailable ({len(available)}):")
        for geoid in available:
            print(f"  ✓ {geoid}")
        
        print(f"\nMissing ({len(missing)}):")
        for geoid in missing:
            print(f"  ✗ {geoid}")
        
        # Suggest alternatives
        if missing:
            print(f"\nSuggested alternatives (first 20 available):")
            alternatives = [geoid for geoid in short_geoids if geoid not in requested_short][:20]
            for geoid in alternatives:
                full_geoid = f"1500000US42003{geoid}"
                print(f"  \"{full_geoid}\",")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_available_blocks() 