function value = random_in_range(range, output_size)
%RANDOM_IN_RANGE - obvious, innit?

if nargin() < 2
    output_size = [1 1];
end

value = range(1) + (range(2) - range(1))*rand(output_size);

end

