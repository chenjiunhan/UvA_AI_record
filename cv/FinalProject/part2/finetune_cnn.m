function [net, info, expdir] = finetune_cnn(varargin)

%% Define options
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('data', ...
  sprintf('cnn_assignment-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = './data/' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb-caltech.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

%opts.train.gpus = [1];



%% update model

net = update_model();

%% TODO: Implement getCaltechIMDB function below

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCaltechIMDB() ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

%%
aa = imdb.images.data(:,:,:,1);
net.meta.classes.name = imdb.meta.classes(:) ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

trainfn = @cnn_train ;
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;

expdir = opts.expDir;
end
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

end

function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

end

% -------------------------------------------------------------------------
function imdb = getCaltechIMDB()
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
classes = {'airplanes', 'cars', 'faces', 'motorbikes'};
splits = {'train', 'test'};

%% TODO: Implement your loop here, to create the data structure described in the assignment

image_root = "../Caltech4/ImageData/";
imaeg_txt_root = "../Caltech4/ImageSets/";

image_paths = ["airplanes_train/" "cars_train/" "faces_train/" "motorbikes_train" ...
               "airplanes_test/" "cars_test/" "faces_test/" "motorbikes_test"];
image_txts = ["airplanes_train.txt" "cars_train.txt" "faces_train.txt" "motorbikes_train.txt" ...
              "airplanes_test.txt" "cars_test.txt" "faces_test.txt" "motorbikes_test.txt"];

image_height = 32;
image_width = 32;
train_set_size = 10; %for each class
test_set_size = 10;
num_classes = 4;

total_txt_list = cell(size(train_set_size * num_classes + test_set_size * num_classes));

for image_txt = image_txts
    index_image_txts = find(image_txts == image_txt);
    txt_list = importdata(strcat(imaeg_txt_root, image_txt));
    
    if ~isempty(contains(image_txt, "train"))
        txt_list = txt_list(1:train_set_size);    
        concatenate_start = ((index_image_txts - 1) * train_set_size + 1);
        concatenate_end = ((index_image_txts) * train_set_size);
    else
        txt_list = txt_list(1:test_set_size);
        concatenate_start = (train_set_size * num_classes + (index_image_txts - 1) * test_set_size + 1);
        concatenate_end = (train_set_size * num_classes + (index_image_txts) * test_set_size);
    end
    
    total_txt_list(concatenate_start:concatenate_end) = txt_list;
end

num_files = length(total_txt_list);   

data = cell(1, num_files);
labels = cell(1, num_files);
sets = cell(1, num_files);

for i=1:num_files
   current_filename = image_root + total_txt_list(i);   
   image_data = im2double(imread(char(current_filename + ".jpg")));
   image_data = imresize(image_data, [image_height, image_width]);
   if i <= train_set_size * num_classes
       label = floor((i - 1) / train_set_size) + 1;
       set = 1;
   else
       label = floor(((i - train_set_size * num_classes) - 1) / test_set_size) + 1;
       set = 2;
   end
   
   data{i} = image_data;
   labels{i} = label;
   sets{i} = set;
end

data = single(cat(4, data{:}));
labels = single(cat(2, labels{:}));
sets = cat(2, sets{:});
%%% subtract mean
dataMean = mean(data(:, :, :, sets == 1), 4);
data = bsxfun(@minus, data, dataMean);

imdb.images.data = data ;

imdb.images.labels = single(labels) ;
imdb.images.set = uint8(sets);
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.classes = classes;
imdb.meta.classes = reshape(classes, [4 1]);

perm = randperm(numel(imdb.images.labels));
imdb.images.data = imdb.images.data(:,:,:, perm);
imdb.images.labels = imdb.images.labels(perm);
imdb.images.set = imdb.images.set(perm);

end
