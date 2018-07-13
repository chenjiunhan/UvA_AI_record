%run('/home/jaqq/vlfeat/vlfeat-0.9.21/toolbox/vl_setup')

image_root = "Caltech4/ImageData/";
imaeg_txt_root = "Caltech4/ImageSets/";

image_paths = ["airplanes_train/" "cars_train/" "faces_train/" "motorbikes_train"];
image_txts = ["airplanes_train.txt" "cars_train.txt" "faces_train.txt" "motorbikes_train.txt"];

train_set_size = 20; %for each class
num_classes = 4;

total_txt_list = cell(size(train_set_size * num_classes));

for image_txt = image_txts
    index_image_txts = find(image_txts == image_txt);
    txt_list = importdata(strcat(imaeg_txt_root, image_txt));    
    txt_list = txt_list(1:train_set_size);
    
    concatenate_start = ((index_image_txts - 1) * train_set_size + 1);
    concatenate_end = ((index_image_txts) * train_set_size);
    
    total_txt_list(concatenate_start:concatenate_end) = txt_list;
end

num_files = length(total_txt_list);   
images = cell(num_files);

for i=1:num_files
   current_filename = image_root + total_txt_list(i);   
   current_image = im2double(imread(char(current_filename + ".jpg")));
   images{i} = current_image;
end

total_txt_list;

% I = images{1};
% I = single(rgb2gray(I)) ;
% [f,d] = vl_sift(I) ;
% size(f)
% size(d)

% d = single(d);
% imshow(I);
% 
% numData = 5000 ;
% dimension = 2 ;
% data = rand(dimension,numData) ;
% 
% numClusters = 30 ;
% [centers, assignments] = vl_kmeans(d, numClusters);
% %size(d)
% %size(data)
% size(centers)
% size(assignments)