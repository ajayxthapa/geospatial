import ee
import geemap
import pandas as pd

# Initialize Earth Engine (authentication handled separately)
ee.Initialize()

def main():
    # 1. Load TIGER counties and filter for Palm Beach County, Florida
    counties = ee.FeatureCollection('TIGER/2018/Counties')
    florida = counties.filter(ee.Filter.eq('STATEFP', '12'))
    palm_beach = florida.filter(ee.Filter.eq('NAME', 'Palm Beach')).first()
    geometry = palm_beach.geometry()
    
    # 2. Load NLCD 1985 land cover data
    nlcd = ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/NLCD_1985').select('landcover')
    
    # 3. Calculate total county area in square miles
    area_image = ee.Image.pixelArea()
    total_area_m2 = area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=30,
        maxPixels=1e12
    ).get('area')
    total_area_sq_miles = ee.Number(total_area_m2).divide(2.59e6)  # Convert m² to sq miles
    
    # 4. Calculate developed (21-24) and barren (31) areas
    developed_mask = nlcd.gte(21).And(nlcd.lte(24))
    barren_mask = nlcd.eq(31)
    
    developed_area_m2 = area_image.updateMask(developed_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=30,
        maxPixels=1e12
    ).get('area')
    
    barren_area_m2 = area_image.updateMask(barren_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geometry,
        scale=30,
        maxPixels=1e12
    ).get('area')
    
    # Convert to square miles
    developed_area_sq_miles = ee.Number(developed_area_m2).divide(2.59e6)
    barren_area_sq_miles = ee.Number(barren_area_m2).divide(2.59e6)
    developed_barren_total = developed_area_sq_miles.add(barren_area_sq_miles)
    
    # 5. Calculate undeveloped area
    undeveloped_area_sq_miles = total_area_sq_miles.subtract(developed_barren_total)
    
    # 6. Prepare and print results
    results = pd.DataFrame({
        'Land Type': ['Total County Area', 'Developed Area', 'Barren Area', 'Developed + Barren', 'Undeveloped Area'],
        'Area (sq miles)': [
            total_area_sq_miles.getInfo(),
            developed_area_sq_miles.getInfo(),
            barren_area_sq_miles.getInfo(),
            developed_barren_total.getInfo(),
            undeveloped_area_sq_miles.getInfo()
        ]
    })
    
    print("Land Area Analysis for Palm Beach County, FL (1985)")
    print("="*50)
    print(results.to_string(index=False))
    print("\nNote: Run in Google Colab after authenticating Earth Engine")

if __name__ == "__main__":
    main()
