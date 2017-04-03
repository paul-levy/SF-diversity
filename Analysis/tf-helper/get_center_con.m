function con = get_center_con(family, contrast)

% hardcoded - based on sfMix as run in 2015/2016 (m657, m658, m660); given
% the stimulus family and contrast level, returns the expected contrast of
% the center frequency.

switch family
    case 1
        if contrast == 1
            con = 1.0000;
        else
            con = 0.3300;
        end
    case 2
        if contrast == 1
            con = 0.6717;
        else
            con = 0.2217;
        end
    case 3
        if contrast == 1
            con = 0.3785;
        else
            con = 0.1249;
        end
    case 4
        if contrast == 1
            con = 0.2161;
        else
            con = 0.0713;
        end
    case 5
        if contrast == 1
            con = 0.1451;
        else
            con = 0.0479;
        end
end            

end

