#!/bin/bash
# Download and extract improved WGAN v2 results

echo "Downloading archive (508 MB)..."
scp -P 13572 -i ~/.ssh/id_ed25519 \
  root@213.173.108.11:/workspace/GANs-for-1D-Signal/results/improved_wgan_v2_all_results.tar.gz \
  ./results/

echo ""
echo "Extracting archive..."
cd results
tar xzf improved_wgan_v2_all_results.tar.gz
rm improved_wgan_v2_all_results.tar.gz

echo ""
echo "✓ Extraction complete!"
echo "Extracted directories:"
ls -d improved_wgan_v2_nz*/ | wc -l
