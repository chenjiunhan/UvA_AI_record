function train_svm(nets, data)

%% replace loss with the classification as we will extract features
nets.pre_trained.layers{end}.type = 'softmax';
nets.fine_tuned.layers{end}.type = 'softmax';

%% extract features and train SVM classifiers, by validating their hyperparameters
[svm.pre_trained.trainset, svm.pre_trained.testset] = get_svm_data(data, nets.pre_trained);
[svm.fine_tuned.trainset,  svm.fine_tuned.testset] = get_svm_data(data, nets.fine_tuned);

%% measure the accuracy of different settings
[nn.accuracy] = get_nn_accuracy(nets.fine_tuned, data);
[svm.pre_trained.predictions, svm.pre_trained.accuracy] = get_predictions(svm.pre_trained);
[svm.fine_tuned.predictions, svm.fine_tuned.accuracy] = get_predictions(svm.fine_tuned);

fprintf('\n\n\n\n\n\n\n\n');

fprintf('CNN: fine_tuned_accuracy: %0.2f, SVM: pre_trained_accuracy: %0.2f, fine_tuned_accuracy: %0.2f\n', nn.accuracy, svm.pre_trained.accuracy(1), svm.fine_tuned.accuracy(1));

end


function [accuracy] = get_nn_accuracy(net, data)

counter = 0;
for i = 1:size(data.images.data, 4)
    
if(data.images.set(i)==2)    
res = vl_simplenn(net, data.images.data(:, :,:, i));

[~, estimclass] = max(res(end).x);

if(estimclass == data.images.labels(i))
    counter = counter+1;
end

end

end

accuracy = counter / nnz(data.images.set==2);
end

function [predictions, accuracy] = get_predictions(data)

best = train(data.trainset.labels, data.trainset.features, '-C -s 0');
model = train(data.trainset.labels, data.trainset.features, sprintf('-c %f -s 0', best(1))); % use the same solver: -s 0
[predictions, accuracy, ~] = predict(data.testset.labels, data.testset.features, model);

end

function [trainset, testset] = get_svm_data(data, net)

trainset.labels = [];
trainset.features = [];

testset.labels = [];
testset.features = [];
size(data.images.data, 4)
for i = 1:size(data.images.data, 4)
    
    res = vl_simplenn(net, data.images.data(:, :,:, i));
    feat = res(end-3).x; feat = squeeze(feat);
    
    if(data.images.set(i) == 1)
        
        trainset.features = [trainset.features feat];
        trainset.labels   = [trainset.labels;  data.images.labels(i)];
        
    else
        
        testset.features = [testset.features feat];
        testset.labels   = [testset.labels;  data.images.labels(i)];
        
        
    end
    
end
labels = double(trainset.labels);
features = double(trainset.features);

test_labels = double(testset.labels);
test_features = double(testset.features);

trainset.labels = double(trainset.labels);
trainset.features = sparse(double(trainset.features'));

testset.labels = double(testset.labels);
testset.features = sparse(double(testset.features'));

% Set parameters
no_dims = 2;
initial_dims = 10;
labels
features
size(labels)
size(features)
perplexity = 30;
% Run t?SNE
mappedX = tsne(features', [], no_dims, initial_dims, perplexity);
% Plot results
gscatter(mappedX(:,1), mappedX(:,2), labels);

% Run t?SNE
mappedX = tsne(test_features', [], no_dims, initial_dims, perplexity);
% Plot results
figure(2), gscatter(mappedX(:,1), mappedX(:,2), test_labels);

end
