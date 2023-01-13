%what is wrong

for i = 1:1200
    for j = 1:3
        if my_labels(i, j) ~= true_labels(i,j)
            disp("Image_:")
            disp(3 * (i-1) + j)
            disp("True value: " + num2str(true_labels(i,j)))
            disp("My value: " + num2str(my_labels(i,j)))
        end
    end
end