rng(0);
patch_sz = 3;
energy_th = 0.95;
total_comp = patch_sz^2;
sigma_n = 5; % std in units of graylevel (0,1,...,255)
img_filename = 'C:\Users\Arie\Downloads\BSR_bsds500\BSR\BSDS500\data\images\train\56028.jpg';

% load eigen-patches
eigpatch_file = sprintf('eigenpatches_%dx%d.mat', patch_sz, patch_sz);
load(eigpatch_file);

% exract central pixels vector from eigen-patches
h = cellfun(@(ep) ep((patch_sz+1)/2, (patch_sz+1)/2), eig_patches);
h = reshape(h,[1 1 total_comp]);

% load image and add noise
I = rgb2gray(imread(img_filename));
In = double(imnoise(I, 'gaussian', 0, (sigma_n/100)^2));
Ir = zeros(size(I));

% filter noisy image with eigen-patches
If = cellfun(@(h) imfilter(In,h), eig_patches, 'UniformOutput', false);
If = cat(3,If{:});

% every pixel is reconstructed with the eigen-patches
% that yield best reconstruction of its patch
[~,inds] = sort(abs(If),3,'descend');
for y=1:size(If,1)
    for x=1:size(If,2)
        pix_coeffs = If(y,x,:);
        v = squeeze(abs(pix_coeffs(:,:,inds(y,x,1:end)))).^2;
        energy = cumsum(v)/sum(v);
        num_comps = find(energy > energy_th, 1, 'first');
        pix_coeffs(:,:,inds(y,x,num_comps+1:end)) = 0;

        Ir(y,x) = sum(pix_coeffs.*h,3);
    end
end

% crop borders
pad = (patch_sz+1)/2;
I  = I(pad:end-pad+1, pad:end-pad+1);
Ir = Ir(pad:end-pad+1, pad:end-pad+1);
In = In(pad:end-pad+1, pad:end-pad+1);

% display results
figure(1); imshow(I, [0 255]);
figure(2); imshow(In,[0 255]);
figure(3); imshow(Ir,[0 255]);
figure(4); imshow(abs(Ir-double(I)),[0 30]);
