rng(0);
patch_sz = 11;
num_comp = 50; % number of components used for the reconstruction
sigma_n = 3; % std in units of graylevel (0,1,...,255)
img_filename = 'C:\Users\Arie\Downloads\BSR_bsds500\BSR\BSDS500\data\images\train\46076.jpg';

% load eigen-patches
eigpatch_file = sprintf('eigenpatches_%dx%d.mat', patch_sz, patch_sz);
load(eigpatch_file);
num_comp = min(num_comp, patch_sz^2);

% exract central pixels vector from eigen-patches
h = cellfun(@(ep) ep((patch_sz+1)/2, (patch_sz+1)/2), eig_patches(1:num_comp));
h = reshape(h,[1 1 num_comp]);

% load image and add noise
I = rgb2gray(imread(img_filename));
In = double(imnoise(I, 'gaussian', 0, (sigma_n/100)^2));

% filter noisy image with eigen-patches
If = cellfun(@(h) imfilter(In,h), eig_patches(1:num_comp), 'UniformOutput', false);
If = cat(3,If{:});
% reconstruct
Ir = convn(If,h(end:-1:1),'valid');

% crop borders
pad = (patch_sz+1)/2;
I  = I(pad:end-pad+1, pad:end-pad+1);
Ir = Ir(pad:end-pad+1, pad:end-pad+1);
In = In(pad:end-pad+1, pad:end-pad+1);

% display results
figure(1); imshow(I, [0 255]);
figure(2); imshow(In,[0 255]);
figure(3); imshow(Ir,[0 255]);
figure(4); imshow(abs(Ir-In),[0 30]);
