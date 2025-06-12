clear;
clc;

img_dir = dir('D:\training_model3_warp_2');
save_path = 'D:\training_model3_crop_2\';
img_dir(1:2)  = [];

hs = 40;
he = 600;
ws = 200;
we = 1000;

for i =1:length(img_dir)
    img_paths = dir(['D:\training_model3_warp_2\', img_dir(i).name]);
    img_paths(1:2) = [];
    for j = 1:length(img_paths)
        img = double(imread([img_paths(j).folder, '\', img_paths(j).name])) / 255.;
        [h, w, c] = size(img);
        temp = zeros(h, w, c);
        temp(hs:he, ws:we, :) = img(hs:he, ws:we, :);
        
        if exist([save_path, num2str(i-1)], 'dir')==0
            mkdir([save_path, num2str(i-1)]);
        end
        imwrite(temp, [save_path, num2str(i-1), '\', img_paths(j).name]);
    end
end


























