# Literature notes for V3 benchmark

Main targets:

1. **BM3D**: K. Dabov, A. Foi, V. Katkovnik, K. Egiazarian, "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering", IEEE TIP, 2007.
2. **NLM**: A. Buades, B. Coll, J.-M. Morel, "A non-local algorithm for image denoising", CVPR, 2005; and the IPOL implementation note "Non-Local Means Denoising", 2011.
3. **DnCNN**: K. Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising", IEEE TIP, 2017.

Use the CSV file for BM3D and DnCNN literature targets at sigma 15, 25, 50. The V3 classical benchmark should be considered complete when BM3D_TAU is close to BM3D literature values, especially on Set12 and BSD68.
