#author: Aron Vukatana
#ML Project: Mapping Railway Networks from Satellite Imagery
#Co: Nicolas Bundscherer, Ethan Ip

import ee
import datetime
import time
import argparse

def create_railway_patches_and_masks(sample_size=10, patch_size=100):
    ''' Creates the export tasks for both raw and masked data
        patch_size -> width covered in meters (and height since its a square)

    '''
    # image collection datasets
    #image = ee.ImageCollection("Switzerland/SWISSIMAGE/orthos/10cm").mosaic()
    image = ee.ImageCollection("Slovakia/orthos/25cm").mosaic()
    points = ee.FeatureCollection('projects/ee-nbundscherer/assets/slovakia')
    rgb_vis = {
        'min': 0,
        'max': 255
    }
    
    def create_centroid_point(feature):
        centroid = feature.geometry().centroid()
        return ee.Feature(centroid)
    
    # creating centroid points, this proved more robust and adapts to polygon aswell
    all_points_from_lines = points.map(create_centroid_point)
    
    seed = int(datetime.datetime.now().timestamp())
    with_random = all_points_from_lines.randomColumn('rand', seed)
    sampled_points = with_random.sort('rand').limit(sample_size)

    def create_square_patch(feature):
        point = feature.geometry()
        half_size = patch_size / 2
        square = point.buffer(half_size).bounds()
        return ee.Feature(square, feature.toDictionary())
    
    square_patches = sampled_points.map(create_square_patch)
    
    #------------------------------------------------------------------------------------
    railway_width = 1.5  # RAILWAY MASKING WIDTH IN METERS
    #-------------------------------------------------------------------------------------
    points_geom_type = points.first().geometry().type().getInfo()
    
    if points_geom_type in ['LineString', 'MultiLineString']:
        print('Using the lines from the coordinate data')
        railway_lines = points
    else:
        print('Creating lines from points (Data type other from LineString/Multistring)')
        points_list = all_points_from_lines.toList(all_points_from_lines.size())
        
        def get_coords(point):
            return ee.Feature(point).geometry().coordinates()
        
        coords = points_list.map(get_coords)
        railway_line = ee.Geometry.LineString(coords)
        railway_lines = ee.FeatureCollection([ee.Feature(railway_line)])
    
    railway_buffers = railway_lines.map(lambda feature: feature.buffer(railway_width))
    
    patch_list = square_patches.toList(square_patches.size())
    count = square_patches.size().getInfo()
    
    export_tasks = []
    
    for i in range(count):
        feature = ee.Feature(patch_list.get(i))
        geom = feature.geometry()
        
        vis_image = image.visualize(
            bands=['R', 'G', 'B'],
            min=0,
            max=255
        )
        
        clipped_image = vis_image.clip(geom)
        
        clipped_image = clipped_image.reproject(crs=image.projection(), scale=0.2) #enforcing 0.2 scale reprojection here
        
        #-----------------------------------RAW EXPORT ---------------------------------
        rgb_task = ee.batch.Export.image.toDrive(
            image=clipped_image,
            description=f'RailwayPatch_{i+6500}', 
            fileNamePrefix=f'railpatch{i+6500}',
            region=geom,
            dimensions='500x500', #enforcing 500x500 dimension here
            maxPixels=1e10,
            fileFormat='GeoTIFF',
            formatOptions={
                'cloudOptimized': True
            }
        )
        
        #--------------------------------------------MASK CREATION----------------------------------
        mask_image = ee.Image().byte().paint(
            featureCollection=railway_buffers,
            color=1
        ).clip(geom).reproject(crs=image.projection(), scale=0.2)
        
        # mask creation
        red_mask_image = ee.Image(1).visualize(
            palette=['FF0000']
        ).updateMask(mask_image)
        
        blended_image = vis_image.blend(red_mask_image).reproject(crs=image.projection(), scale=0.2)
        
        #-------------------------------OVERLAY EXPORT------------------------------------------------------
        overlay_task = ee.batch.Export.image.toDrive(
            image=blended_image,
            description=f'RailwayOverlay_{i+6500}',
            fileNamePrefix=f'railoverlay{i+6500}',
            region=geom,
            dimensions='500x500', 
            maxPixels=1e10,
            fileFormat='GeoTIFF'
        )
 
        export_tasks.extend([rgb_task, overlay_task])
        #----------------------------------------------------------------------------------------------------
    
    return export_tasks

def start_and_monitor_tasks(tasks, max_concurrent=100, check_interval=60):
    """
    Start export tasks with a limit on concurrent tasks and monitor their progress.
    
    Args:
        tasks: List of export tasks to run
        max_concurrent: Maximum number of tasks to run concurrently
        check_interval: Time in seconds between status checks
    """
    active_tasks = []
    completed_tasks = []
    pending_tasks = tasks.copy()
    
    def print_progress():
        total = len(completed_tasks) + len(active_tasks) + len(pending_tasks)
        completed_pct = len(completed_tasks) / total * 100 if total > 0 else 0
        print(f"Progress: [{len(completed_tasks)}/{total}] {completed_pct:.1f}% complete")
        print(f"  - Completed: {len(completed_tasks)}")
        print(f"  - Active: {len(active_tasks)}")
        print(f"  - Pending: {len(pending_tasks)}")
    
    print(f"\nStarting export of {len(tasks)} tasks with max {max_concurrent} concurrent tasks")
    print("\nNote: This process might take A WHILE depending on the number of tasks.")
    
    while pending_tasks or active_tasks:
        still_active = []
        newly_completed = []
        
        for task in active_tasks:
            try:
                status = task.status()['state']
                if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    completed_tasks.append(task)
                    newly_completed.append(task)
                    if status == 'FAILED' and 'error_message' in task.status():
                        print(f"Task failed with error: {task.status()['error_message']}")
                else:
                    still_active.append(task)
            except Exception as e:
                print(f"Error checking task status: {str(e)}")
                still_active.append(task)
        
        for task in newly_completed:
            status = task.status()['state']
            task_name = task.status()['description']
            if status == 'COMPLETED':
                print(f"Completed: {task_name}")
            else:
                print(f"Failed: {task_name} ({status})")
        
        active_tasks = still_active

        newly_started = []
        while len(active_tasks) < max_concurrent and pending_tasks:
            task = pending_tasks.pop(0)
            task.start()
            newly_started.append(task.status()['description'])
            active_tasks.append(task)
        
        if newly_started:
            print(f"Started {len(newly_started)} new tasks: {', '.join(newly_started[:3])}" + 
                  (f" and {len(newly_started)-3} more..." if len(newly_started) > 3 else ""))
        
        if newly_completed or newly_started:
            print_progress()
        
        if active_tasks or pending_tasks:
            time.sleep(check_interval)
    
    print("\n" + "="*50)
    print(f"All {len(completed_tasks)} tasks completed")
    
    #verbose for task status
    success_count = sum(1 for task in completed_tasks if task.status()['state'] == 'COMPLETED')
    fail_count = sum(1 for task in completed_tasks if task.status()['state'] == 'FAILED')
    cancel_count = sum(1 for task in completed_tasks if task.status()['state'] == 'CANCELLED')
    
    print(f"Results:")
    print(f"  - {success_count} successful")
    print(f"  - {fail_count} failed")
    print(f"  - {cancel_count} cancelled")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate railway patches and masks from Earth Engine')
    parser.add_argument('--max_concurrent', type=int, default=100, 
                        help='Maximum number of concurrent export tasks')
    parser.add_argument('--check_interval', type=int, default=60,
                        help='Interval in seconds between task status checks')
    parser.add_argument('--project_id', type=str, default=None,
                        help='Google Cloud project ID to use with Earth Engine')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='Number of sample points to generate')
    parser.add_argument('--patch_size', type=int, default=100,
                        help='Size of each patch in meters')
    args = parser.parse_args()
    
#-----------------------------------INITIALIZING EARTH ENGINE-------------------------------
    ee.Authenticate()

    if args.project_id:
        try:
            ee.Initialize(project=args.project_id)
            print(f"Using project ID: {args.project_id}")
        except Exception as e:
            print(f"Failed to use specified project ID: {str(e)}")
            exit(1)
    else:
        ee.Initialize(project='ee-nbundscherer')
    
    try:
        test = ee.Number(1).getInfo()
        print("Earth Engine connection test successful")
        print("Preparing railway patch and mask export tasks...")
        tasks = create_railway_patches_and_masks(
            sample_size=args.sample_size,
            patch_size=args.patch_size
        )
        print(f"Created {len(tasks)} export tasks")
        
        start_and_monitor_tasks(tasks, max_concurrent=args.max_concurrent, 
                              check_interval=args.check_interval)
    except Exception as e:
        print(f"Error executing Earth Engine operations: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nIf you're seeing authentication errors, try running:")
        print("1. earthengine authenticate")
        print("2. Then run this script with your project ID:")
        print("   python data_extractor.py --project_id=your-project-id")
