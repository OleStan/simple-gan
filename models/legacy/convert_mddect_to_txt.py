import numpy as np
import os

def extract_mddect_signals(npy_file, output_dir, channel=0, classes=None, angles=None, max_samples_per_class=None):
    """
    Extract 1D signals from MDDECT dataset and save as txt files.
    
    Parameters:
    -----------
    npy_file : str
        Path to MDDECT .npy file
    output_dir : str
        Output directory for txt files
    channel : int (0 or 1)
        Which channel to extract (0 or 1)
    classes : list or None
        List of class indices to extract (0-19). None = all classes
    angles : list or None
        List of scanning angle indices to extract (0-7). None = all angles
    max_samples_per_class : int or None
        Maximum samples per class to extract. None = all samples
    """
    
    print(f"Loading MDDECT dataset from {npy_file}...")
    data = np.load(npy_file)
    
    print(f"Dataset shape: {data.shape}")
    print(f"  - Experiments: {data.shape[0]}")
    print(f"  - Scanning angles: {data.shape[1]}")
    print(f"  - Forward/Backward: {data.shape[2]}")
    print(f"  - Repeats: {data.shape[3]}")
    print(f"  - Classes: {data.shape[4]}")
    print(f"  - Temporal points: {data.shape[5]}")
    print(f"  - Channels: {data.shape[6]}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = [
        '2.0mm', '1.9mm', '1.8mm', '1.7mm', '1.6mm', '1.5mm', '1.4mm', '1.3mm',
        '1.2mm', '1.1mm', '1.0mm', '0.9mm', '0.8mm', '0.7mm', '0.6mm', '0.5mm',
        '0.4mm', '0.3mm', 'lift-off', 'normal'
    ]
    
    if classes is None:
        classes = list(range(20))

    if angles is None:
        angles = list(range(data.shape[1]))
    else:
        invalid_angles = [a for a in angles if a < 0 or a >= data.shape[1]]
        if invalid_angles:
            raise ValueError(
                f"Invalid angle indices: {invalid_angles}. Expected values in range 0-{data.shape[1] - 1}"
            )
    
    total_signals = 0
    
    for class_idx in classes:
        class_name = class_names[class_idx]
        print(f"\nProcessing class {class_idx}: {class_name}")
        
        class_signals = []
        
        for exp in range(data.shape[0]):
            for angle in angles:
                for direction in range(data.shape[2]):
                    for repeat in range(data.shape[3]):
                        signal = data[exp, angle, direction, repeat, class_idx, :, channel]
                        class_signals.append(signal)
        
        if max_samples_per_class and len(class_signals) > max_samples_per_class:
            indices = np.random.choice(len(class_signals), max_samples_per_class, replace=False)
            class_signals = [class_signals[i] for i in indices]
            print(f"  Randomly selected {max_samples_per_class} samples from {len(class_signals)} available")
        
        for i, signal in enumerate(class_signals):
            filename = f'mddect_{class_name}_ch{channel}_{i:04d}.txt'
            filepath = os.path.join(output_dir, filename)
            np.savetxt(filepath, signal, fmt='%.6f')
        
        total_signals += len(class_signals)
        print(f"  Saved {len(class_signals)} signals")
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Total signals saved: {total_signals}")
    print(f"Signal length: {data.shape[5]}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    return total_signals, data.shape[5]


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract MDDECT signals to txt files')
    parser.add_argument('--input', type=str, default='./training_data/MDDECT_v1_test.npy',
                        help='Path to MDDECT .npy file')
    parser.add_argument('--output', type=str, default='./data/mddect',
                        help='Output directory')
    parser.add_argument('--channel', type=int, default=0, choices=[0, 1],
                        help='Channel to extract (0 or 1)')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Class indices to extract (0-19). Default: all classes')
    parser.add_argument('--angles', type=int, nargs='+', default=None,
                        help='Scanning angle indices to extract (0-7). Default: all angles')
    parser.add_argument('--max-per-class', type=int, default=None,
                        help='Maximum samples per class. Default: all samples')
    
    args = parser.parse_args()
    
    extract_mddect_signals(
        npy_file=args.input,
        output_dir=args.output,
        channel=args.channel,
        classes=args.classes,
        angles=args.angles,
        max_samples_per_class=args.max_per_class
    )
