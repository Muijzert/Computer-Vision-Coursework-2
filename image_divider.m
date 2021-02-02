function [img1,img2,img3,img4] = image_divider(img, colour)

    switch lower(colour)
        case 'grayscale'
            if(size(img,3) == 3)
                img = rgb2gray(img);
            end
              img = single(img);
            [c, r, ~] = size(img);

            midR = ceil(r/2);
            midC = ceil(c/2);

            img1 = img(midC : end, midR:end);
            img2 = img(midC : end, 1:midR);
            img3 = img(1:midC, midR : end);
            img4 = img(1:midC, 1:midR);
        case 'rgb'
            [c, r, ~] = size(img);
            img = single(img);

            midR = ceil(r/2);
            midC = ceil(c/2);

            img1 = img(midC : end, midR:end, :);
            img2 = img(midC : end, 1:midR, :);
            img3 = img(1:midC, midR : end, :);
            img4 = img(1:midC, 1:midR, :);
    end
end

