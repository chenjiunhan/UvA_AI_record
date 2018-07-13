%% main function 


%% fine-tune cnn
[net, info, expdir] = finetune_cnn();

%% extract trfeatures and train svm

% TODO: Replace the name with the name of your fine-tuned model

file_name = 'fine_tuned_model.mat';
file_path = fullfile(expdir, file_name);
addpath('liblinear/matlab');

if ~exist(file_path, 'file')
    save(file_path, 'net');
end

nets.fine_tuned = load(fullfile(expdir, 'fine_tuned_model.mat')); nets.fine_tuned = nets.fine_tuned.net;
nets.pre_trained = load(fullfile('data', 'pre_trained_model.mat')); nets.pre_trained = nets.pre_trained.net;
data = load(fullfile(expdir, 'imdb-caltech.mat'));

%%
train_svm(nets, data);
