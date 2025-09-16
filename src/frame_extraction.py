import os
import subprocess
import shutil
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import argparse
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import glob
import concurrent.futures 

def convert_png_to_webp(png_path, webp_path, quality=80):
    """
    Convert a single PNG file to WebP format.
    
    Args:
        png_path (str): Path to the source PNG file
        webp_path (str): Path to save the WebP file
        quality (int): WebP quality (0-100), default 80
    
    Returns:
        bool: True if conversion was successful
    """
    try:
        img = Image.open(png_path)
        img.save(webp_path, 'WEBP', quality=quality)
        return True
    except Exception as e:
        print(f"Error converting {png_path}: {e}")
        return False

def compress_directory(input_dir, output_dir, quality=80, max_workers=None):
    """
    Compress all PNG files in a directory to WebP format.
    
    Args:
        input_dir (str): Directory containing PNG files
        output_dir (str): Directory to save WebP files
        quality (int): WebP quality (0-100), default 80
        max_workers (int): Maximum number of worker threads, default is CPU count
    
    Returns:
        dict: Statistics about the conversion process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PNG files in input directory and subdirectories
    png_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                png_path = os.path.join(root, file)
                # Create relative path structure in output directory
                rel_path = os.path.relpath(root, input_dir)
                webp_out_dir = os.path.join(output_dir, rel_path)
                os.makedirs(webp_out_dir, exist_ok=True)
                
                # Create path for webp file
                webp_filename = os.path.splitext(file)[0] + '.webp'
                webp_path = os.path.join(webp_out_dir, webp_filename)
                
                png_files.append((png_path, webp_path))
    
    print(f"Found {len(png_files)} PNG files to convert")
    
    # Convert all PNG files to WebP using thread pool
    successful = 0
    failed = 0
    total_size_before = 0
    total_size_after = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Function to track conversion and collect stats
        def process_file(args):
            png_path, webp_path = args
            success = convert_png_to_webp(png_path, webp_path, quality)
            
            if success:
                png_size = os.path.getsize(png_path)
                webp_size = os.path.getsize(webp_path)
                return success, png_size, webp_size
            return success, 0, 0
        
        # Convert files with progress bar
        results = list(tqdm(
            executor.map(process_file, png_files), 
            total=len(png_files),
            desc="Converting PNG to WebP"
        ))
    
    # Calculate statistics
    for success, png_size, webp_size in results:
        if success:
            successful += 1
            total_size_before += png_size
            total_size_after += webp_size
        else:
            failed += 1
    
    # Prepare statistics
    stats = {
        "total_files": len(png_files),
        "successful": successful,
        "failed": failed,
        "total_size_before_mb": total_size_before / (1024 * 1024),
        "total_size_after_mb": total_size_after / (1024 * 1024),
        "saved_mb": (total_size_before - total_size_after) / (1024 * 1024),
    }
    
    if total_size_before > 0:
        stats["compression_ratio"] = total_size_after / total_size_before
        stats["space_saved_percent"] = (1 - stats["compression_ratio"]) * 100
    
    return stats

def process_video(args):
    """
    Process a single video:
    1. Extract frames as PNG
    2. Convert frames to WebP
    3. Clean up temporary PNG files
    
    Args:
        args: Tuple containing (video_folder, config)
    """
    video_folder, config = args
    
    video_path = os.path.join(config['root_dir'], video_folder, f'{video_folder}.mp4')
    if not os.path.isfile(video_path):
        print(f'Skipping {video_folder}, .mp4 not found')
        return
    
    print(f'Processing {video_folder}...')
    
    # Create temporary directory for PNG frames
    temp_video_dir = os.path.join(config['temp_output_dir'], video_folder)
    os.makedirs(temp_video_dir, exist_ok=True)
    
    # Create output directory for WebP frames
    final_video_dir = os.path.join(config['final_output_dir'], video_folder)
    os.makedirs(final_video_dir, exist_ok=True)
    
    try:
        # Extract frames using ffmpeg
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', "scale='if(gt(iw,ih),512,-2)':'if(gt(iw,ih),-2,512)'",
            os.path.join(temp_video_dir, '%09d.png')
        ]
        
        # Run ffmpeg command
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=True)
        
        # Convert PNG frames to WebP
        stats = compress_directory(
            temp_video_dir, 
            final_video_dir, 
            quality=config['webp_quality'],
            max_workers=config['webp_workers']
        )
        
        # Print statistics for this video
        print(f"\nConversion stats for {video_folder}:")
        print(f"Frames converted: {stats['successful']} (Failed: {stats['failed']})")
        print(f"Size before: {stats['total_size_before_mb']:.2f} MB")
        print(f"Size after: {stats['total_size_after_mb']:.2f} MB")
        print(f"Space saved: {stats['saved_mb']:.2f} MB ({stats.get('space_saved_percent', 0):.1f}%)")
        
        # Cleanup temporary PNG directory if requested
        if config['cleanup_png']:
            print(f"Cleaning up temporary PNG frames for {video_folder}")
            shutil.rmtree(temp_video_dir)
    
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames from {video_folder}: {e}")
    except Exception as e:
        print(f"Error processing {video_folder}: {e}")


def check_and_remove_dir(dir_path):
    """
    Helper function to check if a directory is empty and remove it if it is.
    Returns True if directory was removed, False otherwise.
    """
    try:
        # Check if directory is empty
        files = glob.glob(os.path.join(dir_path, '*'))
        if len(files) == 0:
            os.rmdir(dir_path)
            print(f"Removed empty directory: {dir_path}")
            return True
    except OSError as e:
        print(f"Error processing directory {dir_path}: {e}")
    
    return False

def remove_empty_directories_parallel(root_dir, max_workers=None):
    """
    Efficiently removes empty directories in parallel.
    
    Args:
        root_dir: The root directory to scan for empty subdirectories
        max_workers: Maximum number of worker threads/processes 
                    (None = auto-determined based on CPU count)
    
    Returns:
        Number of directories removed
    """
    removed_dirs = 0
    
    try:
        # Get all immediate subdirectories
        subdirs = glob.glob(os.path.join(root_dir, '*'))

    except OSError as e:
        print(f"Error accessing {root_dir}: {e}")
        return removed_dirs
    
    # Process directories in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all directories for processing and collect futures
        future_to_dir = {executor.submit(check_and_remove_dir, dir_path): dir_path 
                         for dir_path in subdirs}

        for future in concurrent.futures.as_completed(future_to_dir):
            dir_path = future_to_dir[future]
            try:
                if future.result():
                    removed_dirs += 1
            except Exception as e:
                print(f"Exception processing {dir_path}: {e}")
    
    return removed_dirs

def save_cached_temporal_videos(videos, output_dir):
    current_time = datetime.now()

    file_name = current_time.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
    
    videos = '\n'.join(videos)
    # Create and write to the file
    with open(os.path.join(output_dir, file_name), 'w') as file:
        file.write(videos)

    print(f'File saved as {file_name}')

def main():
    parser = argparse.ArgumentParser(description='Extract video frames and convert to WebP')
    
    # Main directories
    parser.add_argument('--root-dir', required=True, help='Root directory containing videos')
    parser.add_argument('--temp-dir', default='TEMP_FRAMES', help='Temporary directory for PNG frames')
    parser.add_argument('--output-dir', required=True, help='Output directory for WebP frames')
    
    # Processing options
    parser.add_argument('--workers', type=int, default=1, help='Number of video processing workers')
    parser.add_argument('--webp-quality', type=int, default=80, help='WebP quality (0-100)')
    parser.add_argument('--webp-workers', type=int, default=None, 
                        help='Number of WebP conversion workers per video')
    parser.add_argument('--no-cleanup', action='store_true', 
                        help='Keep temporary PNG files (default: remove them)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuration dictionary to pass to workers
    config = {
        'root_dir': args.root_dir,
        'temp_output_dir': args.temp_dir,
        'final_output_dir': args.output_dir,
        'webp_quality': args.webp_quality,
        'webp_workers': args.webp_workers,
        'cleanup_png': not args.no_cleanup
    }

    current_temp_dirs = os.listdir(args.temp_dir)

    remove_empty_directories_parallel(args.output_dir, max_workers=args.webp_workers)

    processed_videos = os.listdir(args.output_dir)

    save_cached_temporal_videos(current_temp_dirs, 'frames_processing_cache')

    # Get list of video folders
    video_folders = [d for d in os.listdir(args.root_dir) 
                     if os.path.isdir(os.path.join(args.root_dir, d))]
    
    video_folders = [i for i in video_folders if i not in processed_videos]

    print(f'Found {len(video_folders)} video folders')
    print(f'Using {args.workers} worker(s) for video processing')
    
    # Process videos in parallel
    with Pool(args.workers) as pool:
        process_args = [(video_folder, config) for video_folder in video_folders]
        pool.map(process_video, process_args)
    
    print('All videos processed.')

if __name__ == '__main__':
    main()