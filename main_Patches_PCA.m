% download BSDS500 from: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500
dataset = 'C:\Users\Arie\Downloads\BSR_bsds500\BSR\BSDS500\data\images\train';
N_patches = 10^5;
patch_sz = 3;
rng(0); % always repeat same patches

% collect patches
filelist = dir([dataset, '\*.jpg']);
fl = vertcat({filelist.name});
num_images = length(fl);
samp_per_img = N_patches/num_images;
Patches = zeros(N_patches, patch_sz^2);
Patch_src = cell(num_images,1);

% sample patches from images
patch_idx = 1;
for img_idx = 1:num_images
    fprintf('sampling patches from image %d/%d\n', img_idx, num_images);
    filename = fullfile(dataset, fl{img_idx});
    I = rgb2gray(imread(filename));
    
    % randomization limits
    H = size(I,1);
    W = size(I,2);
    y_max = H-patch_sz+1;
    x_max = W-patch_sz+1;
    
    % randomize top-left coordinate
    x0 = ceil(x_max*rand(samp_per_img,1));
    y0 = ceil(y_max*rand(samp_per_img,1));
    
    % store patch
    im_patches = arrayfun(@(x,y) imcrop(I, [x y patch_sz-1 patch_sz-1]), x0, y0, 'UniformOutput', false);
    im_patches = cellfun(@(patch) patch(:)', im_patches, 'UniformOutput', false);
    Patches(patch_idx:patch_idx+samp_per_img-1, :) = vertcat(im_patches{:});
    
    % store patch position
    Patch_src{img_idx}.filename = fl{img_idx};
    Patch_src{img_idx}.rects = [x0 y0 repmat([patch_sz patch_sz], [samp_per_img 1])];
    
    patch_idx = patch_idx+samp_per_img;
end

% PCA patches
patch_cov = cov(Patches);
[eig_vecs, lambdas] = pcacov(patch_cov);

% visualize eigen-patches
eig_vecs_cell = num2cell(eig_vecs, 1);
eig_patches = cellfun(@(eig_vec) reshape(eig_vec, [patch_sz patch_sz]), eig_vecs_cell, 'UniformOutput', false);

% display all eigen-patches
figure(1);
for eig_idx = 1:length(eig_patches)
    subplot(patch_sz,patch_sz,eig_idx);
    imshow(eig_patches{eig_idx}, []);
    
    title_txt = sprintf('\\lambda=%.1f', lambdas(eig_idx));
    title(title_txt);
end

% display power spectrum
figure(2);
plot(1:length(eig_patches), lambdas);
xlabel('component number');
ylabel('eigen value');

% save results & settings
mat_filename = sprintf('eigenpatches_%dx%d.mat', patch_sz, patch_sz);
save(mat_filename, 'patch_sz', 'eig_patches', 'lambdas', 'N_patches', 'fl', 'Patch_src');
