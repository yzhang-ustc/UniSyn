import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# res_folder = '/Users/yue/Documents/code/UniSyn/1to1_ours_results/adjust_lambda/brats_11_t1t1c_batch16_5dataset_100l1_datainfo50/images/t1c'
# res_folder = '/Users/yue/Documents/code/UniSyn_git/results/adjust_lambda/brats_11_t1t1c_best/images/t1c'
res_folder = '/Users/yue/Documents/code/UniSyn_git/results/adjust_lambda/brats_11_t2flair_best/images/flair'
gt_root = '/Users/yue/Documents/data/BraTS19/png/new_data/brats19/brats19 2/test/flair'


def load_image(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img)
        return img_array.astype(np.float64) / 255.0
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def compute_metrics(img1, img2):
    try:
        # Ensure image dimensions are consistent
        if img1.shape != img2.shape:
            # If dimensions are inconsistent, resize img2 to img1's dimensions
            img2_resized = Image.fromarray((img2 * 255).astype(np.uint8)).resize((img1.shape[1], img1.shape[0]))
            img2 = np.array(img2_resized).astype(np.float64) / 255.0
        
        # Compute PSNR
        psnr_value = psnr(img1, img2, data_range=1.0)
        
        # Compute SSIM
        ssim_value = ssim(img1, img2, data_range=1.0)
        
        return psnr_value, ssim_value
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None, None


def get_case_name(filename):
    return filename.replace(filename.split('_')[-1],'')


def get_slice_number(filename):
    """Extract slice number from filename: the number in split('_')[-1] after removing '.png'"""
    name_without_ext = filename.replace('.png', '').replace('.PNG', '')
    last_part = name_without_ext.split('_')[-1]
    try:
        return int(last_part)
    except ValueError:
        return 0


def stack_case_volume(folder, case_files):
    """Sort all images of a case by slice number and stack them into a volume"""
    # Sort by slice number
    sorted_files = sorted(case_files, key=get_slice_number)
    
    slices = []
    for filename in sorted_files:
        img_path = os.path.join(folder, filename)
        img = load_image(img_path)
        if img is not None:
            slices.append(img)
        else:
            print(f"Warning: Failed to load {filename}, skipping")
    
    if len(slices) == 0:
        return None
    
    # Stack into 3D volume
    volume = np.stack(slices, axis=0)
    return volume


def compute_volume_metrics(vol1, vol2):
    """Compute PSNR and SSIM of two volumes"""
    try:
        # Ensure volume dimensions are consistent
        if vol1.shape != vol2.shape:
            print(f"Warning: Volume shapes mismatch: {vol1.shape} vs {vol2.shape}")
            # Resize vol2 to vol1's dimensions
            min_depth = min(vol1.shape[0], vol2.shape[0])
            min_height = min(vol1.shape[1], vol2.shape[1])
            min_width = min(vol1.shape[2], vol2.shape[2])
            vol1 = vol1[:min_depth, :min_height, :min_width]
            vol2 = vol2[:min_depth, :min_height, :min_width]
        
        # Compute PSNR
        psnr_value = psnr(vol1, vol2, data_range=1.0)
        
        # Compute SSIM
        ssim_value = ssim(vol1, vol2, data_range=1.0)
        
        return psnr_value, ssim_value
    except Exception as e:
        print(f"Error computing volume metrics: {e}")
        return None, None


def compute_folder_metrics(folder1, folder2):
    """Compute PSNR and SSIM for all corresponding PNG images in two folders by case volume"""
    if not os.path.exists(folder1):
        print(f"Error: Folder 1 does not exist: {folder1}")
        return
    
    if not os.path.exists(folder2):
        print(f"Error: Folder 2 does not exist: {folder2}")
        return
    
    # Get all PNG files in both folders
    png_files1 = {f for f in os.listdir(folder1) if f.lower().endswith('.png')}
    png_files2 = {f for f in os.listdir(folder2) if f.lower().endswith('.png')}
    
    # Find files that exist in both folders
    common_files = png_files1 & png_files2
    
    if len(common_files) == 0:
        print("No common PNG files found in both folders")
        return
    
    print(f"Found {len(common_files)} common PNG files")
    
    # Group by case name
    cases_dict = {}
    for filename in common_files:
        case_name = get_case_name(filename)
        if case_name not in cases_dict:
            cases_dict[case_name] = []
        cases_dict[case_name].append(filename)
    
    print(f"Found {len(cases_dict)} cases")
    
    psnr_values = []
    ssim_values = []
    processed_count = 0
    
    # Compute metrics for each case
    for case_name in sorted(cases_dict.keys()):
        case_files = cases_dict[case_name]
        print(f"Processing case: {case_name} ({len(case_files)} slices)")
        
        # Stack into volume
        vol1 = stack_case_volume(folder1, case_files)
        vol2 = stack_case_volume(folder2, case_files)
        
        if vol1 is None or vol2 is None:
            print(f"Skipping case {case_name}: Failed to stack volumes")
            continue
        
        # Compute metrics
        psnr_val, ssim_val = compute_volume_metrics(vol1, vol2)
        
        if psnr_val is not None and ssim_val is not None:
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            processed_count += 1
            print(f"  Case {case_name}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")
        else:
            print(f"Skipping case {case_name}: Failed to compute metrics")
    
    # Compute mean and standard deviation
    if len(psnr_values) > 0 and len(ssim_values) > 0:
        avg_psnr = np.mean(psnr_values)
        std_psnr = np.std(psnr_values)
        avg_ssim = np.mean(ssim_values)
        std_ssim = np.std(ssim_values)
        
        print(f"\n=== Results ===")
        print(f"Processed cases: {processed_count}/{len(cases_dict)}")
        print(f"PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
        print(f"SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        
        return {
            'avg_psnr': avg_psnr,
            'std_psnr': std_psnr,
            'avg_ssim': avg_ssim,
            'std_ssim': std_ssim,
            'processed_count': processed_count,
            'total_count': len(cases_dict)
        }
    else:
        print("No valid metrics computed")
        return None


if __name__ == "__main__":
    compute_folder_metrics(res_folder, gt_root)

