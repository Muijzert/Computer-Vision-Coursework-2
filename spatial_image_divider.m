function [image_feats] = spatial_image_divider(img,colour)
step_size = 3;
bin_size = 2;

switch lower(colour)
    case 'grayscale'
        image_feats = [];
        
        [imgDiv{1}, imgDiv{2}, imgDiv{3}, imgDiv{4}] = image_divider(img, colour);

        for j = 1:4
            [~, SIFT_feature{j}] = vl_dsift(imgDiv{j},'fast','Size',bin_size,'Step',step_size); 
        end

        for i=1:4
           image_feats = [image_feats, SIFT_feature{i}];
        end
    case 'rgb'
        image_feats = [];
        
        [imgDiv{1}, imgDiv{2}, imgDiv{3}, imgDiv{4}] = image_divider(img, colour);
        
        for j = 1:4
            [~, SIFT_feature{j}] = vl_phow(imgDiv{j},'fast','True','step',step_size,'sizes',bin_size,'color', colour);
        end

        for i=1:4
           image_feats = [image_feats, SIFT_feature{i}];
        end
    end
end

