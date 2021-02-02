function [image_feats] = get_spatial_pyramids(image_paths,k, colour)

load('vocab_s.mat');
vocab_size = size(vocab, 1); 
step_size = 3;
bin_size = 2;


%loop for every image
switch lower(colour)
    case 'grayscale'
        for i=1:size(image_paths,1)
            total_features = [];

            img = imread(image_paths{i});
            [imgDiv{1}, imgDiv{2}, imgDiv{3}, imgDiv{4}] = image_divider(img, colour);
            img = single(rgb2gray(img));
            for p=1:4
                SIFT_feature{p} = spatial_image_divider(imgDiv{p}, colour);
            end
            
            SIFT_feature{5} = spatial_image_divider(img, colour);
            
            [~, SIFT_feature{6}] = vl_dsift(img,'fast','Size',bin_size,'Step',step_size);

            for j=1:6
               total_features =  [total_features, SIFT_feature{j}];
            end
            total_features = single(total_features);
            [clusterNo, ~] = knnsearch(vocab', total_features', 'k', k);
            [image_feats(i,:), ~] = histcounts(clusterNo, vocab_size);
            image_feats(i,:) = rescale(image_feats(i,:));
        end
    case 'rgb'
        for i=1:size(image_paths,1)
            total_features = [];

            img = imread(image_paths{i});

            
            [imgDiv{1}, imgDiv{2}, imgDiv{3}, imgDiv{4}] = image_divider(img, colour);
            img = single(img);
            for p=1:4
                SIFT_feature{p} = spatial_image_divider(imgDiv{p}, colour);
            end
            
            SIFT_feature{5} = spatial_image_divider(img, colour);

            [~, SIFT_feature{6}] = vl_phow(img,'fast','True','step',step_size,'sizes',bin_size,'color', colour);

            for j=1:6
               total_features =  [total_features, SIFT_feature{j}];
            end
            total_features = single(total_features);
            [clusterNo, ~] = knnsearch(vocab', total_features', 'k', k);
            [image_feats(i,:), ~] = histcounts(clusterNo, vocab_size);
            image_feats(i,:) = rescale(image_feats(i,:));
        end
end
end

